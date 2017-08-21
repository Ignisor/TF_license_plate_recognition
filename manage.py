import sys
import logging

from abc import ABCMeta, abstractmethod


class Command(metaclass=ABCMeta):
    """Base class for creating commands"""
    # possible names to call command
    names = []
    description = ''

    @abstractmethod
    def handle(self, *args):
        """abstract method that called when command executes"""
        pass

    @staticmethod
    def get_arg(index, err_message, *args):
        try:
            return args[index]
        except IndexError:
            raise AttributeError(err_message)


class Help(Command):
    names = ['help', '?', 'h']
    description = 'List all commands'

    def handle(self, *args):
        print('Commands: ')
        for command in all_subclasses(Command):
            print(f'{" / ".join(command.names)} - {command.description}')


class Train(Command):
    names = ['train', ]
    description = 'Train model. Usage "train [model_name] [iterations]".'

    def handle(self, *args):
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)

        model_name = Command.get_arg(0, 'You must specify model name as first positional argument', *args)
        iterations = int(args[1]) if len(args) > 1 else 1000

        model_class = self.import_(model_name)
        model = model_class()

        model.train(iterations)

    def import_(self, name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod


class RunUrl(Command):
    names = ['run_url', ]
    description = 'Run neural network. Usage "run_url [model_name] [url]".'

    def handle(self, *args):
        model_name = Command.get_arg(0, 'You must specify model name as first positional argument', *args)
        url = Command.get_arg(1, 'You must specify url as second positional argument', *args)

        model_class = self.import_(model_name)
        model = model_class()

        print(model.run_single(url))

    def import_(self, name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod


class TempCmd(Command):
    names = ['tmp', ]
    description = 'Temporary command used to run some code. Like shell.'

    def handle(self, *args):
        # import logging
        # logger = logging.getLogger()
        # logger.setLevel(logging.DEBUG)
        #
        # from lp_finder import LPRecogniser
        # from base import VideoProcessor
        # vp = VideoProcessor('', LPRecogniser, 4 / 1)
        #
        # import time
        # t = time.time()
        # result = vp.process_video('/home/ignisor/Downloads/pl111.avi')
        # logging.debug(f"video processed in: {time.time() - t:.4f}s")

        import cv2
        img = cv2.imread('/home/ignisor/dev/tf_lpr/tf_lpr/lp_chars_recognition/test_lp.jpg')

        from lp_chars_recognition import LPProcessor
        lp_proc = LPProcessor()
        # lp = lp_proc.crop(img)
        chars = lp_proc.split(img)


def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in all_subclasses(s)]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise AttributeError('No command provided. Use "help" to get list of commands.')

    input_cmd = sys.argv[1].lower()

    # find and execute command
    for command in all_subclasses(Command):
        if input_cmd in command.names:
            command().handle(*sys.argv[2:])
            break
