import atexit
import os
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import logging

Gst.init(None)

logging.basicConfig(
    level=logging.DEBUG,
)

class CameraRecorder():
    def __init__(self, width = 224, height= 224, fps= 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.recording = False
        self.gst_pipeline = None

        self.logger = logging.getLogger("CameraRecorder")
        self.logger.setLevel(logging.DEBUG)

        atexit.register(self.stop)

    def _gst_str(self, target_dir):
        assert os.path.isdir(target_dir), "Target directory {} does not exist".format(target_dir)

        location = os.path.join(target_dir, 'frame_%05d.jpg')
        pipeline_str = "nvarguscamerasrc sensor-mode=3 ! video/x-raw(memory:NVMM), width={}, height={}, framerate=30/1 ! ".format(self.width, self.height) + "videorate ! video/x-raw(memory:NVMM), framerate={}/1 ! ".format(self.fps) + "nvjpegenc ! multifilesink location={}".format(location)

        print(pipeline_str)

        self.logger.info("GStreamer pipeline string: {}".format(pipeline_str))
        return pipeline_str

    def start_recording_jpegs(self, target_dir):
        assert os.path.isdir(target_dir), "Target directory {} does not exist".format(target_dir)

        if self.recording:
            self.logger.error("Pipeline is already running.")
            return

        pipeline_string = self._gst_str(target_dir)
        self.gst_pipeline = Gst.parse_launch(pipeline_string)

        bus = self.gst_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.gst_pipeline.set_state(Gst.State.PLAYING)
        self.recording = True
        self.logger.info("Started recording to: {}".format(target_dir))

        self.loop = GLib.MainLoop()
        try:
            self.logger.info("Camera recording in progress... Press Ctrl+C to stop.")
            self.loop.run()
        except KeyboardInterrupt:
            self.logger.info("Stopping camera recording...")
        finally:
            self.stop()


    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.logger.info("End-of-stream reached.")
            self.stop()

        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("GStreamer Error: {}\nDebug Info: {}".format(err, debug))
            self.stop()

    def stop(self):
        if self.gst_pipeline and self.recording:
            self.gst_pipeline.send_event(Gst.Event.new_eos())

            time.sleep(0.5)

            self.gst_pipeline.set_state(Gst.State.NULL)
            self.recording = False
            self.logger.info("Recording stopped.")




if __name__ == "__main__":
    output_directory = "./test_ouput"
    os.makedirs(output_directory, exist_ok=True)

    recorder = CameraRecorder(width=224, height=224, fps=1)

    recorder.start_recording_jpegs(output_directory)
