from picamera2 import Picamera2,Preview
from picamera2.encoders import H264Encoder
import time


awb = 0
ae = 0
gain = 8.0
framerate=30
exp = 33333
path = '/dev/shm/'


picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (1200, 800)},
                                             raw={'format': 'SRGGB10', 'size': (1332, 990)})
picam2.configure(config)

picam2.set_controls({"AwbEnable": awb, "AeEnable": ae,'AnalogueGain':gain,'FrameRate':framerate,'ExposureTime':exp})

picam2.start_preview(Preview.QTGL, x=100, y=100, width=640, height=480) 

encoder = H264Encoder(qp=30)

picam2.start_recording(encoder, f'{path}test.h264',pts=f'{path}timestamp.txt')

time.sleep(10)

picam2.stop_recording()
