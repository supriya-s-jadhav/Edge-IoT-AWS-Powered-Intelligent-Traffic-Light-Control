# Simple image recognition/classifier based on the Detectron 2
# Runnable on EC2, if AMI is Deep Learning AMI (Ubuntu 18.04) Version 35.0 (or the one with pre-installed pytorch 1.6.0, and CUDA)
#   ami-0c322afdce03ef272
# Underlying CNN model is Mask R-CNN, based on ResNet-50 and Feature Pyramide Network 3x

# Prerequisites
import argparse
import multiprocessing as mp
import numpy as np
import torch, torchvision, detectron2
import os, json, cv2, random, tqdm, time, boto3
import warnings

# Imports for the basic Detectron2 setup
# UserWarning coming from the detectron2 is suppressed (Nothing I can do about it right now)
warnings.simplefilter("ignore", UserWarning)
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

# From the detectron 2 predictor class
class Visualize(object):
    class IOMetadata:
        """
        Args:
            imageHandle ():
            videoHandle (cv.VideoCapture):
            basename ():
            outputFile ():
            outputAnalysis ():
        """ 
        def __init__(self, imgHandle = None, vidHandle = cv2.VideoCapture(), 
                     basename = "", outputFile = "", outputAnalysis = ""):
            #self._imageHandle = imgHandle
            self._videoHandle = vidHandle
            self._basename = basename
            self._outputFile = outputFile
            self._outputAnalysis = outputAnalysis
        
        @property
        def VideoHandle(self):
            return self._videoHandle
        
        @VideoHandle.setter
        def VideoHandle(self, value):
            self._videoHandle = value
        
        @property
        def Basename(self):
            return self._basename
        
        @Basename.setter
        def Basename(self, value):
            self._basename = value
        
        @property
        def OutputFile(self):
            return self._outputFile
        
        @OutputFile.setter
        def OutputFile(self, value):
            self._outputFile = value
        
        @property
        def OutputAnalysis(self):
            return self._outputAnalysis
        
        @OutputAnalysis.setter
        def OutputAnalysis(self, value):
            self._outputAnalysis = value
        
        def SetIOPath(self, inputFilePath):
            assert (os.path.isfile(inputFilePath)), "Error: input file {} non existent!".format(inputFilePath)
            self._basename = inputFilePath
            
            # Set the handle for the video
            self._videoHandle = cv2.VideoCapture(self._basename)
            
            # Save the image with prediction boxs and score            
            head_tail = os.path.splitext(self._basename)
            self._outputFile = head_tail[0] + "_output" + head_tail[1]
            self._outputAnalysis = head_tail[0] + "_analysis.csv"
            
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """ 
        
        self.IOMetadata = self.IOMetadata()
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
    
        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video, currentFrame = None):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        
        # Timeframe
        fps = self.IOMetadata.VideoHandle.get(cv2.CAP_PROP_FPS)
        timestamps = [self.IOMetadata.VideoHandle.get(cv2.CAP_PROP_POS_MSEC)]
        calc_timestamps = [0.0]

        # Parallel-processing enabled (Multi-GPU setting)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    #self.get_predicted_objects_list(predictions, "video")
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                #self.get_predicted_objects_list(predictions, "video")
                yield process_predictions(frame, predictions)
                
        # Parallel-processing disabled (default)
        else:
            CurrentFrameNumber = 0;
            
            for frame in frame_gen:
                timestamps.append(self.IOMetadata.VideoHandle.get(cv2.CAP_PROP_POS_MSEC))
                #calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
                
                predictions = self.predictor(frame)
                
                # Create two dictionaries and append
                metadataDict = {"Frame Index": CurrentFrameNumber, "Timestamp": timestamps[CurrentFrameNumber]}
                occurrenceDict = self.transform_predictions_to_dict(predictions, 
                                                                    CurrentFrameNumber, 
                                                                    EnableVerbose=False)
                metadataDict.update(occurrenceDict)
                
                append_occurrence_data ( metadataDict, self.IOMetadata.OutputAnalysis )
                yield process_predictions(frame, predictions)

                CurrentFrameNumber = CurrentFrameNumber + 1

    def transform_predictions_to_dict(self, predictions, frame_index=None, EnableVerbose=False):
        CurrentMetadataLabels = self.metadata.thing_classes
        ObjectsList = [CurrentMetadataLabels[elem] 
                       for elem in predictions["instances"].pred_classes.data.tolist()]
        CountDict = dict()
        
        for i in ObjectsList:
          CountDict[i] = CountDict.get(i, 0) + 1
        
        if EnableVerbose:
            print("Frame #: {}, Predictions: {}".format(
                    frame_index,
                    CountDict
                ))
        
        return CountDict

def setup_cfg(args):
    from detectron2 import model_zoo

    # load config from file and command-line arguments
    cfg = get_cfg()

    # Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    # Default is the COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
    #if args.video_input:
    #    assert args.input is None, "You cannot specify the video input and input at the same time!"
    #    args.config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    #    args.confidence_threshold = 0.6
    #    print (args.config_file)

    cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
    #cfg.merge_from_list(args.opts)

    # Set score_threshold for builtin models
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config_file)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        
    cfg.freeze()
    return cfg

def setup_interactive_cfg(mode = None):
    from detectron2 import model_zoo
    TESTBED_CNN_MODEL_CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    TESTBED_CNN_MODEL_CONFIDENCE_SCORE = 0.5
    
    # load config from file and command-line arguments
    cfg = get_cfg()
    
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(TESTBED_CNN_MODEL_CONFIG_FILE))

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(TESTBED_CNN_MODEL_CONFIG_FILE)
    
    # Set manually the score_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = TESTBED_CNN_MODEL_CONFIDENCE_SCORE  # set threshold for this model

    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")

    parser.add_argument(
        "--config-file",
        default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--video-input", help="Path to video file.")
    
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; or input.jpg",
    )
    
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will store the results into output.jpg",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--archive2aws",
        "-a",
        action="store_true",
        help="Optional argument for storing the results to the AWS S3 Bucket",
    )

    parser.add_argument(
        "--destination",
        type=str,
        default="sshan-ml",
        help="Name of the destination S3 bucket",
    )

    parser.add_argument(
        "--destkey",
        type=str,
        default="3MPOC",
        help="Key of the destination S3 bucket",
    )

    return parser

def send_to_s3 ( filePath, bucketName, keyName ):
    s3_connection = boto3.client('s3')
    fileName = os.path.basename(filePath)

    with open(filePath, "rb") as f:
        s3_connection.upload_fileobj(f, bucketName, keyName + '/' + fileName)
    
    print("File {} has been successfully transferred to S3. Destination: {}, Key: {}".format(
        fileName, bucketName, keyName))

def save_occurrence_data ( occurrenceDict, fileName ):
    import csv
    csv_columns = ['Objects', 'Occurrence']

    try:
        with open(fileName, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for keyValue in occurrenceDict:
                writer.writerow({'Objects': keyValue, 'Occurrence': occurrenceDict[keyValue]})
    except IOError:
        print("Encountered an unexpected I/O error")
        
def append_occurrence_data ( occurrenceDict, fileName ):
    try:
        with open(fileName, 'a') as csvfile:
            csvfile.write(', '.join("%s : %s" % (k, v) for k,v in occurrenceDict.items()))
            csvfile.write('\n')
            
    except IOError:
        print("Encountered an unexpected I/O error")

def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    # 1. import some common detectron2 utilities
    from detectron2.utils.logger import setup_logger
    from detectron2.data.detection_utils import read_image
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog

    # 2. Setup detectron2 logger
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # Create a configuration
    cfg = setup_cfg(args)
    demo = Visualize(cfg)

    # Specify the key for the S3 'folder'; 
    output_s3_key = args.destkey

    # Begin of if; image, video or cam
    if args.input:
        # Initialize the predictor with the configuration
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()

            # Predict
            predictions, visualized_output = demo.run_on_image(img)
            occurrences_dict = demo.transform_predictions_to_dict(predictions)

            # Log the assessment
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            # Save the image with prediction boxs and score
            head_tail = os.path.splitext(path)

            if args.output is None:
                output_file = head_tail[0] + "_output" + head_tail[1]

            else:
                output_file = args.output

            # Save the analysis file
            output_s3_key += "/images"
            output_analysis_csv = head_tail[0] + "_analysis.csv"

            visualized_output.save(output_file)
            save_occurrence_data( occurrences_dict, output_analysis_csv )

            demo.IOMetadata.OutputFile = output_file
            demo.IOMetadata.OutputAnalysis = output_analysis_csv
    
    elif args.video_input:
        #assert len(args.input) == 0, "You cannot specify the video input and image input, at the same time"
        demo.IOMetadata.SetIOPath(args.video_input)

        width = int(demo.IOMetadata.VideoHandle.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(demo.IOMetadata.VideoHandle.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = demo.IOMetadata.VideoHandle.get(cv2.CAP_PROP_FPS)
        num_frames = int(demo.IOMetadata.VideoHandle.get(cv2.CAP_PROP_FRAME_COUNT))

        output_file = cv2.VideoWriter(
            filename=demo.IOMetadata.OutputFile,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            # fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )

        #assert os.path.isfile(args.video_input), "You cannot do batch-processing for video segmentation!"
        #assert args.output is not None, "Output must be specified!"

        for vis_frame in tqdm.tqdm(demo.run_on_video(demo.IOMetadata.VideoHandle), total=num_frames):
            if demo.IOMetadata.OutputFile:
                output_file.write(vis_frame)
            else:
                print("This version doesn't support the cv2 window")

        # Release the inputVideoHandle, as it is done automatically by OpenCV destructor
        demo.IOMetadata.VideoHandle.release()
        output_file.release()
        
        # Set the S3 bucket tag for archival
        output_s3_key += "/videos"
    
    else:
        print("Encountered an unknown error!")

    # End of if
    # Send the output file to the S3
    if args.archive2aws:
        send_to_s3(demo.IOMetadata.OutputFile, bucketName=args.destination, keyName=output_s3_key)
        send_to_s3(demo.IOMetadata.OutputAnalysis, bucketName=args.destination, keyName=output_s3_key)

if __name__== "__main__":
    main()