#
# Copyright (C) Stanislaw Adaszewski, 2017
#

from argparse import ArgumentParser
from PIL import Image
import numpy as np
import re
import glob
import os


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('dname', type=str)
    parser.add_argument('first', type=int)
    parser.add_argument('last', type=int)
    parser.add_argument('--ext', type=str, default='JPG')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    glob_expr = os.path.join(args.dname, '*.%s' % args.ext)
    print 'glob_expr:', glob_expr
    fnames = glob.glob(glob_expr)
    model_name = os.path.split(fnames[0])[1]
    print 'Sample matching file name:', model_name
    model_name = os.path.splitext(model_name)[0]
    m = re.match('([a-zA-Z_]*)([0-9]*)([a-zA-Z_]*)', model_name)
    prefix = m.group(1)
    padding = len(m.group(2))
    suffix = m.group(3)
    print 'prefix:', prefix, 'suffix:', suffix
    print 'zero padding numbers to:', padding, 'characters'
    fmt = '%s%%0%dd%s.%s' % (prefix, padding, suffix, args.ext)
    print 'fmt:', fmt
    sum_ = None
    cnt = 0
    for i in range(args.first, args.last+1):
        im_fname = fmt % i
        print 'im_fname:', im_fname
        im = Image.open(os.path.join(args.dname, im_fname))
        im = np.array(im)
        if sum_ is None:
            sum_ = np.zeros(im.shape, dtype=np.float64)
        sum_ += im
        cnt += 1
    print 'cnt:', cnt
    sum_ /= cnt
    sum_[sum_ > 255] = 255
    sum_[sum_ < 0] = 0
    res_im = Image.fromarray(sum_.astype(np.uint8))
    res_fmt = '%s%%0%dd_%%0%dd%s_mean.%s' % (prefix, padding, padding, suffix, args.ext)
    res_fname = res_fmt % (args.first, args.last)
    print 'res_fname:', res_fname
    res_im.save(res_fname)
    print 'Done.'


if __name__ == '__main__':
    main()
