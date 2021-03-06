import os
import sys

import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn.model import *
import colorsys
import random
import filterpy.kalman as kl
from scipy.optimize import linear_sum_assignment
from trcnn.utils import keepClasses, sample_boxes, squarify, bbs, gating, box_center, IoU, normalize_boxes
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import uuid
import numpy as np
np.random.seed(1)
from trcnn.utils import best_buddies_assignment

# Import measurement for tracking 
from measurements import  save_instances 


class DART():
    '''
    Deep Appearance Robust Tracker
    '''

    def __init__(self, config, output_file, class_names, log_file):

        open(output_file, 'w').close()
        open(log_file, 'w').close()
        self.trackf = open(output_file, 'a')
        self.lgf = open(log_file, 'a')
        self.track_id = 0
        self.obj_list = []
        self.lost_obj = []
        self.config = config
        self.class_names = class_names


    def initialize(self, r, feat_sets, pyr_levels, image, frame=0):


        # include configuration parameters for object
        for i in range(len(r['class_ids'])):
            self.obj_list += [trackedObject(self.track_id, r['masks'][:,:,i], r['rois'][i,:],
                              r['class_ids'][i], feat_sets[i], pyr_levels[i],
                              kalman=[self.config.KF_P, self.config.KF_Q, self.config.KF_R])]
            self.track_id += 1

        # Prediction Step   
        for obj in self.obj_list:
            obj.location_prediction()

        if self.config.SAVE_DETECTIONS:
            # save first frame
            save_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                self.class_names, ids = [str(x.id)[:4] for x in self.obj_list], 
                                file_name = str(frame)+'.png', colors=[x.color for x in self.obj_list], show_bbox=False)


        self.lgf.write("Frame {}\n".format(frame))

        # write first frame track results
        for obj in self.obj_list:
            if obj.tracking_state == 'New' or obj.tracking_state == 'Tracked':
                self.trackf.write("{} {} {} 0 0 -10.0 {} {} {} {} -1000.0 -1000.0 -1000.0 -10.0 -1 -1 -1 {}\n".format(
                            frame, obj.id, self.class_names[obj.class_name], 
                            float(obj.bbox[1]), float(obj.bbox[0]), float(obj.bbox[3]), float(obj.bbox[2]), 1))

        return self.obj_list, self.lost_obj, self.track_id

    def associate(self, r, feat_sets, pyr_levels, image, frame):

        print("Frame {}".format(frame)) 

        self.lgf.write("Frame {}\n".format(frame))

        # save 3-sigma ellipses images
        if self.config.USE_SPATIAL_CONSTRAINTS and self.config.SAVE_SPATIAL_CONSTRAINTS:
            fig = plt.figure() 
            ax1 = fig.add_subplot(111, aspect='equal')
            ax1.imshow(image)
            for obj in self.obj_list:
                b_gain = np.sqrt((obj.bbox[2]-obj.bbox[0])**2 + (obj.bbox[3]-obj.bbox[1])**2)/np.sqrt(2)/2
                x = obj.x_minus
                P = obj.P_minus
                lambda_, v = np.linalg.eig(P[:2,:2])
                lambda_ = np.sqrt(lambda_)
                j = 3
                ell = Ellipse(xy=(x[0], x[1]),
                    width=(lambda_[0]*j+b_gain)*2, height=(lambda_[1]*j+b_gain)*2,
                    angle=np.rad2deg(np.arccos(v[0, 0])), fill=False,lw=3)
                ell.set_facecolor('none')
                ax1.add_patch(ell)
            fig.canvas.flush_events()
            plt.draw()
            plt.savefig('_'+str(frame)+'.png')
            # plt.show()
            ax1.cla()

        # for each newly found object initialize trackedObject
        temp_list = []
        temp_scores = []
        for i in range(len(r['class_ids'])):

            # initialize tracked Objects for this current frame
            temp_list += [trackedObject(uuid.uuid4(), r['masks'][:,:,i], r['rois'][i,:],
                            r['class_ids'][i], feat_sets[i], pyr_levels[i],
                            kalman=[self.config.KF_P, self.config.KF_Q, self.config.KF_R])]
            temp_scores += [r['scores'][i]]


        # Compare old and new objects' appearance 
        # gating condition
        buddy_list = []
        peek_matrix = np.zeros((len(self.obj_list),len(temp_list)))
        for i,obj in enumerate(self.obj_list):
            buddy_list_i = []
            for j,temp in enumerate(temp_list):

                if self.config.USE_SPATIAL_CONSTRAINTS:

                    b_gain = np.sqrt((obj.bbox[2]-obj.bbox[0])**2 + (obj.bbox[3]-obj.bbox[1])**2)/np.sqrt(2)/2
                    if gating(obj.x_minus, obj.P_minus, temp.x, b_gain):
                        peek_matrix[i,j] = 100
                        buddy_list_i += [None]
                    else:
                        bb_sim, bb_b = bbs(obj,temp)
                        peek_matrix[i,j] = 1-bb_sim
                        buddy_list_i += [bb_b]
                else:
                    bb_sim, bb_b = bbs(obj,temp)
                    peek_matrix[i,j] = 1-bb_sim
                    buddy_list_i += [bb_b]


            buddy_list += [buddy_list_i]

        # pad cost matrix if more old objects than new objects
        if peek_matrix.shape[0] > peek_matrix.shape[1]:
            peek_matrix = squarify(peek_matrix, 100)

        print('\n'+str(peek_matrix))


        # run assignment (appearance model)
        # row_ind, col_ind = linear_sum_assignment(peek_matrix)
        row_ind, col_ind = best_buddies_assignment(peek_matrix)


        # log matches
        matching_scores = []
        for i in row_ind:
            matching_scores += [peek_matrix[i,col_ind[i]]]
        self.lgf.write(np.array_str(peek_matrix, max_line_width=1000)+'\n'+str(row_ind)+'\n'+str(col_ind)+'\n'+str(matching_scores)+'\n')

        num_new = len(temp_list)
        temp_matched = [False]*num_new

        # update objects
        # propagate previous objects in new frame
        # also save pairs of bounding boxes
        # pairs = []
        for i,obj in enumerate(self.obj_list):
            j = col_ind[i]
            # if there is a match (old>temp)
            if j < num_new and peek_matrix[i,col_ind[i]] <= self.config.MATCH_THRESHOLD:
                # refress data
                obj.refresh_state(True, window=self.config.FRAME_THRESHOLD)
                obj.mask = temp_list[j].mask

                # pairs += [buddy_list[i][j]]

                obj.refress_encoding(temp_list[j].encoding, buddy_list[i][j], 
                    c_old = self.config.APP_DRIFT_MULTIPLIER)
                obj.class_name = temp_list[j].class_name
                temp_matched[j] = True
                obj.score = 1 - peek_matrix[i,j]
                obj.scores += [[frame, 1 - peek_matrix[i,j]]]
                obj.location_update(obj.x_minus, obj.P_minus, temp_list[j].bbox )


            # if there is no match, this object is occluded in this frame (or lost if it 
            # is occluded for more than N frames)
            else:
                obj.refresh_state(False, window=self.config.FRAME_THRESHOLD)
                obj.location_update(obj.x_minus, obj.P_minus, None )

        # patches += [pairs]
        # initialize new objects
        # det_thresh = 0.7
        for i, temp in enumerate(temp_list):
            # for new object initialization the detection score should be >= det_thresh
            if not temp_matched[i]: # and temp_scores[i] >= det_thresh:
                temp.id = self.track_id
                temp.score = temp_scores[i]
                temp.scores += [[frame, temp_scores[i]]]
                self.track_id += 1
                self.obj_list += [temp]

        # keep objects appeared in current frame
        obj_list_fr = [x for x in self.obj_list if x.tracking_state=='Tracked' or x.tracking_state=='New']
        num_obj = len(obj_list_fr)

        # Prepare object data for saving image
        boxes = np.empty([num_obj,4])
        masks = np.empty([image.shape[0], image.shape[1], num_obj])
        for i in range(num_obj):
            boxes[i,:] = obj_list_fr[i].bbox
            masks[:,:,i] = obj_list_fr[i].mask

        # save current frame with found objects
        if self.config.SAVE_DETECTIONS:
            save_instances(image, boxes, masks, [-1 for x in obj_list_fr], 
                            self.class_names, ids = [str(x.id)[:4]+' '+'{:.2f}'.format(x.score) for x in obj_list_fr], 
                            file_name = str(frame).zfill(6)+'.png',colors=[x.color for x in obj_list_fr],show_bbox=False)

        for obj in self.obj_list:
            if obj.tracking_state == 'New' or obj.tracking_state == 'Tracked':
                self.trackf.write("{} {} {} 0 0 -10.0 {} {} {} {} -1000.0 -1000.0 -1000.0 -10.0 -1 -1 -1 {}\n".format(
                            frame, obj.id, self.class_names[obj.class_name], 
                            float(obj.bbox[1]), float(obj.bbox[0]), float(obj.bbox[3]), float(obj.bbox[2]), 1))

        # remove lost objects
        self.lost_obj += [x for x in self.obj_list if x.tracking_state=='Lost']
        self.obj_list = [x for x in self.obj_list if x.tracking_state!='Lost']

        # Prediction Step   
        for obj in self.obj_list:
            obj.location_prediction()

# TrackRCNN adds functionality to MaskRCNN for tracking 
class TrackRCNN(modellib.MaskRCNN):

    # Overriding build function to include feature maps in output
    # TODOs: Merge this with the appearance encoding class
    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            shape=[None, None, 3], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
            tracking_anchors = KL.Input(shape=[None, 4], name="tracking_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                             stage5=True, train_bn=config.TRAIN_BN)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads

            # Preprocess track rois and merge them with rpn rois
            track_rois = TrackRoisLayer(
                proposal_count=100,
                nms_threshold=config.RPN_NMS_THRESHOLD,
                name="TROI",
                config=config)([tracking_anchors])
            rpn_rois = KL.Concatenate(axis = 1)([rpn_rois, track_rois])

            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in 
            # normalized coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            #TODO: add appearance encoding functionality here
            model = KM.Model([input_image, input_image_meta, input_anchors, tracking_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox,P2,P3,P4,P5],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    # Overwrite detect function to output feat maps, image metas, molded images and 
    # normalized coordinates of boxes (detections)
    def detect(self, images, tracking_anchors, classes_det, class_names, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)

        tracking_anchors = normalize_boxes(tracking_anchors, images[0].shape[:2])

        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        tracking_anchors = np.broadcast_to(tracking_anchors, (self.config.BATCH_SIZE,) + tracking_anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _, P2,P3,P4,P5=\
            self.keras_model.predict([molded_images, image_metas, anchors, tracking_anchors], verbose=0)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "fp_maps": [P2, P3, P4, P5],
                "metas": image_metas,
                "images": molded_images,
                "detections": detections
            })

        r = results[0]
        r = keepClasses(r, classes_det, class_names)
        return r

class TrackRoisLayer(KE.Layer):
    """Preprocess tracking anchors

    Inputs:
        tracking_anchors: [batch, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        corrected and padded tracking anchors in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(TrackRoisLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):

        anchors = inputs[0]

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(anchors,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # pad proposals
        def pad(boxes):
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(boxes)[0], 0)
            boxes = tf.pad(boxes, [(0, padding), (0, 0)])
            return boxes

        proposals = utils.batch_slice(boxes, pad,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)

class RoiAppearance():

    """RoiAppearance
    Appearance pyramid features for detected bounding boxes

    """
    def __init__(self, config):
        """config: A Sub-class of the Config class
        """
        self.config = config
        self.keras_model = self.build(config=config)

    def build(self, config):
        """Build PyramidROIAlign architecture.
            input_rois: [batch, num_rois, [y1, x1, y2, x2]]
            config: A Sub-class of the Config class
            input_image_meta: meta information from input image
            input_feat_{i}: features from i pyramid layer
        """

        # Inputs
        input_rois = KL.Input(
            shape=[ None, 4], name="input_rois")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        input_feat_2 = KL.Input(shape=[  256, 256, 256], 
            name='input_P2')
        input_feat_3 = KL.Input(shape=[  128, 128, 256], 
            name='input_P3')
        input_feat_4 = KL.Input(shape=[   64,  64, 256], 
            name='input_P4')
        input_feat_5 = KL.Input(shape=[   32,  32, 256], 
            name='input_P5')

        # Graph definition 
        pool_size = config.POOL_SIZE
        pooled = PyramidROIAlign([pool_size, pool_size],
                name="roi_align_classifier")(
                [input_rois, input_image_meta, input_feat_2,
                 input_feat_3, input_feat_4, input_feat_5])

        model = KM.Model([input_rois, input_image_meta, input_feat_2,
                            input_feat_3, input_feat_4, input_feat_5],
                        [pooled], name='roi_appear'
                )

        return model


    def rois_encode(self, rois, meta, f2, f3, f4, f5):
        """ Encode detection boxes using feature pyramid 
            rois: [batch, num_rois, [y1, x1, y2, x2]]
            meta: meta information from input image
            f{i}: features from i pyramid layer
        """

        pools = self.keras_model.predict([rois, meta, f2, f3, f4, f5])
        return pools

class ParticleDescription():

    def __init__(self, config):
        self.config = config
        self.encoder = RoiAppearance(config=self.config)
    def encode(self, r, image):

        # Compute particle bounding boxes and pyramid levels
        pyr_levels, bboxes_batch, split_list, image = sample_boxes(r, image=image)
       
        # Keep a copy of absolute coordinates
        bboxes_abs_batch = bboxes_batch.copy()

        # Normalize coordinates for encoding
        bboxes_batch = np.array([normalize_boxes(bboxes_batch, image.shape[:2])])

        # Encode particle boxes
        app = self.encoder.rois_encode(bboxes_batch,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
                    r['fp_maps'][2],r['fp_maps'][3])
        # app = self.encoder.rois_encode(bboxes_batch,r['metas'],r['fp_maps'][0],r['fp_maps'][0],
        #     r['fp_maps'][0],r['fp_maps'][0])
        app_list = [app[0,i,:,:,:].flatten('F') for i in range(app.shape[1])]

        app = np.array(app_list)

        # append features to feature list
        # st_i = 1 because bboxes_abs_batch first row is dummy (zero initialization)
        st_i = 1
        feat_sets = []
        for split in split_list:

            feat_sets += [[bboxes_abs_batch[st_i:split+st_i,:], \
                    app[st_i:split+st_i,:]/np.linalg.norm(app[st_i:split+st_i,:], axis = 1)[:,None],\
                    np.ones(len(app[st_i:split+st_i,:]))]]
            st_i += split


        return feat_sets, pyr_levels

class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]
        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        # Quick hack to select only from level 2
        # minimum(5,...) 
        roi_level = tf.minimum(2, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 4)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )

class trackedObject():

    def __init__(self, ID, mask, bbox, class_name, encodings, pyramid, 
                kalman = [np.diag([10,10,1000,1000]),np.diag([1,1,10,10]),np.diag([100,100])], 
                color = None):

        self.id = ID 
        self.mask = mask
        self.bbox = bbox
        self.class_name = class_name
        self.tracking_state = 'New'
        self.encoding = encodings
        self.color = color if color is not None else self.init_color()
        self._occluded_cnt = 0
        self.pyramid = pyramid
        self.score = 0
        self.smooth_traj = True
        self.scores = []
        self.dim_x = 4
        self.dim_o = 2
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.P = kalman[0]
        self.Q = kalman[1]
        self.R = kalman[2]
        self.x = np.array([0,0,0,0]).reshape((4,1))
        self.x_minus = np.array([0,0,0,0]).reshape((4,1))
        self.x[:2] = box_center(self.bbox)
        self.x_minus[:2] = box_center(self.bbox)
        self.P_minus = np.diag([10,10,1000,1000])

    def init_color(self):
        N = 20
        brightness = 1.0
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        return random.choice(colors)

    def in_frame(self, shape):
        if self.bbox[0]<0 or self.bbox[1]<0 or self.bbox[2]< 0 or self.bbox[3]< 0:
            return False
        if self.bbox[0] > shape[0] or self.bbox[2] > shape[0]:
            return False
        if self.bbox[1] > shape[1] or self.bbox[3] > shape[1]:
            return False
        return True

    def location_prediction(self):
        self.x_minus, self.P_minus = kl.predict(self.x, self.P, self.F, self.Q)

    def location_update(self, x, P, z, R = None):
        # if there is a detection 
        if z is not None:
            # use exact bounding box, refress uncertainty
            self.bbox = z
            self.x[:2] = box_center(z)
            x_obs, self.P = kl.update(x, P, box_center(z), self.R, self.H)
            self.x[2:] = x_obs[2:]
        # if there is no detection 
        else:
            # use predicted bounding box center and uncertainty
            delta_x = self.x_minus - self.x
            self.bbox = [self.bbox[0]+delta_x[1], self.bbox[1]+delta_x[0], 
                         self.bbox[2]+delta_x[1], self.bbox[3]+delta_x[0]]
            self.x = self.x_minus
            self.P = self.P_minus

    def refresh_state(self, matched, window=5):

        # TODO: include xc, yc, h, w, velocities state
        if self.tracking_state == 'New':
            if matched: 
                self.tracking_state = 'Tracked'
            else:
                self.tracking_state = 'Occluded'
                self._occluded_cnt = 1
        elif self.tracking_state == 'Tracked':
            if not matched:
                self.tracking_state = 'Occluded'
                self._occluded_cnt = 1
        elif self.tracking_state == 'Occluded':
            if matched:
                self.tracking_state = 'Tracked'
                self._occluded_cnt = 0
            else:
                self._occluded_cnt += 1
                if self._occluded_cnt > window: # Frames being occluded
                    self.tracking_state = 'Lost'



    def refress_encoding(self, particles, buddy_list, c_old = 0.8):
        
        # encoding consists of two arrays
        # a card(particles)x dim(feature_vec) array of features for each particle
        # a card(particles)x 4 array of bounding boxes for each particle 
        # [feat_array, box_array]
        num_particles = max(len(particles[0]), len(self.encoding[0]))

        N = len(particles[0]) + len(self.encoding[0])

        self.encoding[2] *= c_old
        
        probs = np.hstack((particles[2],self.encoding[2]))
        p = probs/np.sum(probs)

        enc_all = [np.vstack((particles[0],self.encoding[0])), np.vstack((particles[1],self.encoding[1])), probs ] 
        point_val = np.random.choice(np.array(list(range(0,N))), size = num_particles, p=p , replace = False)

        new_enc = [enc_all[0][point_val], enc_all[1][point_val], probs[point_val]]

        self.encoding = new_enc





