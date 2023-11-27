import argparse
import tensorflow as tf
import warnings 
warnings.filterwarnings("ignore")
from train import train
from test import test
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

parser = argparse.ArgumentParser()
parser.add_argument("--model_EMU", help="Model to train", default="comb",choices=["MobileNetV2","ResNet50","comb","None"])
parser.add_argument("--model_ANOMALY", help="Model to train", default="comb",choices=["IF","AE","comb"])
parser.add_argument("--feature_extractor", help="Feature extractor", default="comb",choices=["ResNet50","MobileNetV2","comb"])
parser.add_argument("--window_size", help="Window size", default=16, type=int,choices=[8,16,33,64])
parser.add_argument("--step_size", help="Step size", default=8, type=int,choices=[8,16,33,64])
parser.add_argument("--batch_size", help="Batch size", default=2048, type=int)
parser.add_argument("--mode", help="Mode", default="train",choices=["train","test"])
args = parser.parse_args()
args.window_size = (args.window_size,args.window_size,3)
if args.model_EMU == "comb":
    args.model_EMU = ["MobileNetV2","ResNet50"]
else:
    args.model_EMU = [args.model_EMU]
if args.mode == "train":
    training = train(model_EMU = args.model_EMU,model_ANOMALY = args.model_ANOMALY,window_size=args.window_size, step_size=args.step_size, batch_size=args.batch_size)
    if training:
        print("Training completed successfully!")
    else:
        print("Training failed!")
else:
    print("Testing...")
    test(model_EMU = args.model_EMU,model_ANOMALY = args.model_ANOMALY,window_size=args.window_size, step_size=args.step_size, batch_size=args.batch_size)
    print("Testing completed successfully!")
