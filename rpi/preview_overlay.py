from picamera2 import Picamera2, Preview
import settings_loader
from time import sleep
import numpy as np
from libcamera import Transform





picam2 = Picamera2()
picam2.start_preview(Preview.QTGL,transform=Transform(90))
preview_config = picam2.create_preview_configuration(main={"size": (1456, 1088)},raw={'format':'R10','size':(1456,1088)})
picam2.configure(preview_config)

picam2.set_controls({ "AeEnable": 0,"ExposureTime": 222400,"AnalogueGain": 15.67,'FrameRate':50.0})
overlay = np.zeros((1088, 1456, 4), dtype=np.uint8)
overlay[:, 725:730] = (255, 0, 0, 255)
overlay[542:546, :] = (255, 0, 0, 255)
overlay[:,1452:1456] = (255, 0, 0, 255)

picam2.set_overlay(overlay)

picam2.start()
sleep(2000)
