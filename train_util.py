import glob
import os
import re
from pathlib import Path
from types import SimpleNamespace

import click


class Args(SimpleNamespace):
    @property
    def batch_gpu(self):
        return self.__dict__['batch-gpu']

    @batch_gpu.setter
    def batch_gpu(self, value):
        self.__dict__['batch-gpu'] = value

    @property
    def snap_img(self):
        return self.__dict__['snap-img']

    @snap_img.setter
    def snap_img(self, value):
        self.__dict__['snap-img'] = value

    @property
    def snap_kimg(self):
        return self.__dict__['snap-kimg']

    @snap_kimg.setter
    def snap_kimg(self, value):
        self.__dict__['snap-kimg'] = value

    @property
    def snap_img_kimg(self):
        return self.__dict__['snap-img-kimg']

    @snap_img_kimg.setter
    def snap_img_kimg(self, value):
        self.__dict__['snap-img-kimg'] = value

    @property
    def save_latest(self):
        return self.__dict__['save-latest']

    @save_latest.setter
    def save_latest(self, value):
        self.__dict__['save-latest'] = value

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def build_command(self, background: bool = True):
        cmd = ''
        cmd += 'nohup' if background else ''
        cmd += ' python train.py'
        for k, v in self.__dict__.items():
            if v is None:
                continue
            elif type(v) == bool:
                cmd += f' --{k}' if v else ''
            else:
                cmd += f' --{k}={v}'
        cmd += ' > /dev/null 2>&1 &' if background else ''
        return cmd

    def get_resume(self):
        outdir = self.outdir

        rkimg = 0
        resume = None
        for path in glob.glob(os.path.join(outdir, '*/*.pkl')):
            match = re.search(r'network-snapshot-(\d{6})', path)

            if not match:
                continue

            kimg = int(match.group(1))

            if kimg > rkimg:
                rkimg = kimg
                resume = path

        self.rkimg = rkimg or self.rkimg
        self.resume = resume or self.resume

@click.command()
@click.argument('background', type=bool, default=False)
def main(background=False):
    workspace = os.getcwd()
    path = Path(__file__).absolute()
    os.chdir(path.parent)
    os.environ['TORCH_EXTENSIONS_DIR'] = str(path.parent.joinpath('.cache/torch_extensions'))

    args = Args()
    args.cfg = 'stylegan2'
    args.outdir = os.path.join(workspace, 'training')
    args.data = os.path.join(workspace, 'data/512x512.zip')
    args.workers = 4
    args.gpus = 4
    args.batch = 32
    args.batch_gpu = 8
    args.gamma = 8
    args.snap = 100
    args.snap_img = 50
    # args.snap_kimg = 1000
    # args.snap_img_kimg = 200
    args.save_latest = True
    args.kimg = 25000
    args.metrics = 'fid50k_full' # none fid50k_full
    args.resume = ''
    args.rkimg = 0

    args.get_resume()

    os.system(args.build_command(background))

if __name__ == '__main__':
    main()
