import argparse
import multiprocessing
import os
import cv2
import numpy as np
from PIL import Image
import skvideo.io
import scipy.misc
from multiprocessing import Pool

class OpticalFlowExtractor:
    def __init__(self, args):
        self.input_path = args.input
        self.output_dir = args.outputDir
        self.algorithm = args.algorithm
        self.bound = args.bound
        self.class_folder = args.classFolder
        self.force = args.force
        self.help_msg = args.help_msg
        self.input_frames = args.inputFrames
        self.new_height = args.newHeight
        self.new_short = args.newShort
        self.new_width = args.newWidth
        self.step = args.step
        self.save_type = args.saveType
        self.verbose = args.verbose

    def extract_process(self,video_name, video_path, label):

                try:
                    videocapture = skvideo.io.vread(video_path).astype(cp.float32)
                except Exception as e:
                    print(f'{video_path} read error: {str(e)}')
                    return

                print(video_path)

                if videocapture.sum() == 0:
                    print(f'Could not initialize capturing {video_path}')
                    return

                len_frame = len(videocapture)
                frame_num = 0
                image, prev_image, gray, prev_gray = None, None, None, None
                num0 = 0

                while True:
                    if num0 >= len_frame:
                        break

                    frame = videocapture[num0]
                    num0 += 1

                    if frame_num == 0:
                        image = np.zeros_like(frame)
                        gray = np.zeros_like(frame)
                        prev_gray = np.zeros_like(frame)
                        prev_image = frame
                        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
                        frame_num += 1

                        # to pass the out of stepped frames
                        step_t = self.step
                        while step_t > 1:
                            num0 += 1
                            step_t -= 1
                        continue

                    image = frame
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    frame_0 = prev_gray
                    frame_1 = gray

                    flowDTVL1 = self.calculate_optical_flow(frame_0, frame_1)
                    self.save_flows(flowDTVL1, image, self.output_dir, frame_num, self.bound, video_name, label)
                    prev_gray = gray
                    prev_image = image
                    frame_num += 1

                    # to pass the out of stepped frames
                    step_t = self.step
                    while step_t > 1:
                        num0 += 1
                        step_t -= 1

    def extract_optical_flow(self):
        pool = Pool(processes=multiprocessing.cpu_count())

        for root, dirs, files in os.walk(self.input_path):
            for video_name in files:
                original, label = os.path.split(root)
                video_path = os.path.join(root, video_name)

                # Sử dụng pool để xử lý nhiều video đồng thời
                pool.apply_async(self.extract_process, zip(video_name,video_path, label))

        pool.close()
        pool.join()

    def calculate_optical_flow(self, frame_0, frame_1):
        frame_0 = cp.asarray(frame_0)
        frame_1 = cp.asarray(frame_1)
        dtvl1 = cv2.cuda_DualTVL1OpticalFlow.create()
        flowDTVL1 = dtvl1.calc(frame_0, frame_1, None)
        return cp.asnumpy(flowDTVL1)
    def save_flows(self, flows, image, save_dir, num, bound, video_name, label):
        flow_x = self.to_img(flows[..., 0], bound)
        flow_y = self.to_img(flows[..., 1], bound)

        label_dir = os.path.join(save_dir, label, video_name)

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        save_img = os.path.join(label_dir, f'img_{num:05d}.jpg')
        scipy.misc.imsave(save_img, image)

        save_x = os.path.join(label_dir, f'flow_x_{num:05d}.jpg')
        save_y = os.path.join(label_dir, f'flow_y_{num:05d}.jpg')

        flow_x_img = Image.fromarray(flow_x)
        flow_y_img = Image.fromarray(flow_y)

        scipy.misc.imsave(save_x, flow_x_img)
        scipy.misc.imsave(save_y, flow_y_img)

    def to_img(self, raw_flow, bound):
        flow = raw_flow
        flow[flow > bound] = bound
        flow[flow < -bound] = -bound
        flow -= -bound
        flow *= (255 / float(2 * bound))
        return flow

    @staticmethod
    def get_video_list(videos_root):
        video_list = []
        for cls_names in os.listdir(videos_root):
            cls_path = os.path.join(videos_root, cls_names)
            for video_ in os.listdir(cls_path):
                video_list.append(cls_names+'$label$'+video_)
        video_list.sort()
        return video_list, len(video_list)


def parse_args():
    parser = argparse.ArgumentParser(description="Dense Optical Flow Extraction")

    parser.add_argument("input", help="Filename of video or folder of frames or a list.txt")
    parser.add_argument("-a", "--algorithm", default="tvl1", help="Optical flow algorithm (nv/tvl1/farn/brox)")
    parser.add_argument("-b", "--bound", type=int, default=32, help="Maximum of optical flow")
    parser.add_argument("--cf", "--classFolder", action="store_true", help="OutputDir/class/video/flow.jpg")
    parser.add_argument("-f", "--force", action="store_true", help="Regardless of the marked .done file")
    parser.add_argument("-h", "--help_msg", action="store_true", help="Print help message")
    parser.add_argument("--if", "--inputFrames", action="store_true", help="Inputs are frames")
    parser.add_argument("--newHeight", "--nh", type=int, default=0, help="New height")
    parser.add_argument("--newShort", "--ns", type=int, default=0, help="Short side length")
    parser.add_argument("--newWidth", "--nw", type=int, default=0, help="New width")
    parser.add_argument("-o", "--outputDir", default=".", help="Root dir of output")
    parser.add_argument("-s", "--step", type=int, default=0, help="Right - left (0 for img, non-0 for flow)")
    parser.add_argument("--saveType", "--st", default="jpg", help="Save format type (png/h5/jpg)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dense Optical Flow Extraction")
    # # ... (same as before)
    #
    args = parser.parse_args()

    if args.help_msg:
        parser.print_help()
    else:
        # data_root = "./mmaction2/tools/data/hmdb51/videos/"
        # videos_root = os.path.join(data_root)
        # new_dir = "./mmaction2/tools/data/hmdb51/rawframes"
        # Get the list of video names
        # video_list, num_videos = OpticalFlowExtractor.get_video_list(videos_root)

        # Create an instance of OpticalFlowExtractor
        optical_flow_extractor = OpticalFlowExtractor(args)
        optical_flow_extractor.extract_optical_flow()
        # Iterate through each video and extract optical flow
        # for video_name in tqdm(video_list, desc="Extracting Optical Flow"):
        #     # Update the input path in the extractor
        #
        #     optical_flow_extractor.input_path = video_name.split('$label$')[1]
        #     optical_flow_extractor.class_folder = video_name.split('$label$')[0]
        #     optical_flow_extractor.extract_optical_flow()
            # Set your actual directories here
