import threading
import queue
import time
import os
import png
import numpy
import math


DAVIS_PALETTE_4BIT = [[  0,   0,   0],
                      [128,   0,   0],
                      [  0, 128,   0],
                      [128, 128,   0],
                      [  0,   0, 128],
                      [128,   0, 128],
                      [  0, 128, 128],
                      [128, 128, 128],
                      [ 64,   0,   0],
                      [191,   0,   0],
                      [ 64, 128,   0],
                      [191, 128,   0],
                      [ 64,   0, 128],
                      [191,   0, 128],
                      [ 64, 128, 128],
                      [191, 128, 128]]


class ReadSaveImage(object):
    def __init__(self):
        super(ReadSaveImage, self).__init__()

    def check_path(self, fullpath):
        path, filename = os.path.split(fullpath)
        if not os.path.exists(path):
            os.makedirs(path)


class ReadSaveDAVISChallengeLabels(ReadSaveImage):
    def __init__(self, bpalette=DAVIS_PALETTE_4BIT, palette=None):
        super(ReadSaveDAVISChallengeLabels, self).__init__()
        self._palette = palette
        self._bpalette = bpalette
        self._width = 0
        self._height = 0

    @property
    def palette(self):
        return self._palette

    def save(self, image, path):
        self.check_path(path)
        if self._palette is None:
            palette = self._bpalette
        else:
            palette = self._palette
        bitdepth = int(math.log(len(palette))/math.log(2))
        height, width = image.shape
        file = open(path, 'wb')
        writer = png.Writer(width, height, palette=palette, bitdepth=bitdepth)
        writer.write(file, image)

    def read(self, path):
        try:
            reader = png.Reader(path)
            width, height, data, meta = reader.read()
            if self._palette is None:
                self._palette = meta['palette']
            image = numpy.vstack(data)
            self._height, self._width = image.shape
        except png.FormatError:
            image = numpy.zeros((self._height, self._width))
            self.save(image, path)
        return image


class ImageSaveHelper(threading.Thread):
    def __init__(self):
        super(ImageSaveHelper, self).__init__()
        self._alive = True
        self._queue = queue.Queue(2 ** 20)
        self.start()

    @property
    def alive(self):
        return self._alive

    @alive.setter
    def alive(self, alive):
        self._alive = alive

    @property
    def queue(self):
        return self._queue

    def kill(self):
        self._alive = False

    def enqueue(self, datatuple):
        ret = True
        try:
            self._queue.put(datatuple, block=False)
        except queue.Full:
            print('ImageSaveHelper - enqueue full')
            ret = False
        return ret

    def run(self):
        while True:
            while not self._queue.empty():
                args, method = self._queue.get(block=False, timeout=2)
                method.save(*args)
                self._queue.task_done()
            if not self._alive and self._queue.empty():
                break
            time.sleep(0.0001)
