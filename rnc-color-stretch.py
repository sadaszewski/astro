#
# Copyright (C) Stanislaw Adaszewski, 2017
#
# Based on rnc-color-stretch (for Davinci) by
# Roger N. Clark:
# http://www.clarkvision.com/articles/astrophotography-rnc-color-stretch/
#

from argparse import ArgumentParser
from PIL import Image
import numpy as np


def create_parser():
	parser = ArgumentParser()
	parser.add_argument('imfile', type=str)
	parser.add_argument('--rootpower', type=float, default=6.0)
	parser.add_argument('--rootiter', type=int, default=1)
	parser.add_argument('--rootpower2', type=float, default=1.0)
	parser.add_argument('--pcntclip', type=float, default=0.005)
	parser.add_argument('--nocolorcorrect', action='store_false')
	parser.add_argument('--colorenhance', type=float, default=1.0)
	parser.add_argument('--tonecurve', action='store_true')
	parser.add_argument('--cumstretch', action='store_true')
	parser.add_argument('--skylevelfactor', type=float, default=0.06)
	# parser.add_argument('--rgbskylevel', type=float, default=1024.0)
	parser.add_argument('--rgbskylevelrs', type=float, default=4096.0)
	parser.add_argument('--zerosky', type=float, default=[4096.0, 4096.0, 4096.0], nargs=3)
	# parser.add_argument('--zeroskyred', type=float, default=4096.0)
	# parser.add_argument('--zeroskygreen', type=float, default=4096.0)
	# parser.add_argument('--zeroskyblue', type=float, default=4096.0)
	parser.add_argument('--scurve', type=int, default=0)
	parser.add_argument('--setmin', type=float, nargs=3)
	# parser.add_argument('--setminr', type=float, default=0.0)
	# parser.add_argument('--setming', type=float, default=0.0)
	# parser.add_argument('--setminb', type=float, default=0.0)
	return parser
	
	
def find_clip_lvls(ahist, npixm):
	alow = np.zeros(ahist.shape[0])
	
	ahistcum = np.cumsum(ahist, axis=1)
	for i in range(0, ahist.shape[0]):
		idx = np.where(ahistcum[i, :]>=npixm)[0]
		print('clip level for chan %d: %f' % (i, idx[0]))
		if idx[0] > 39900:
			print('clipping to 39900')
			alow[i] = 39900
		else:
			alow[i] = idx[0]
	
	return alow


def subtract_sky(im, npass, args):

	skylevelfactor = args.skylevelfactor
	zerosky = args.zerosky

	c = np.array(im)
	
	for pass_ in range(0, npass):
		print('pass_:', pass_)
		chist = np.zeros((65536, c.shape[2]), dtype=np.float64)
		for i in range(0, c.shape[2]):
			chist[:, i] = np.histogram(c[:, :, i], np.arange(0, 65537))[0]
			
		chistsm = np.zeros(chist.shape) # smoothed
		ism = 300
		
		for i in range(c.shape[2]):
			chistsm[:, i] = np.convolve(chist[:, i], np.ones(ism)/ism, mode='same')
			
		chistargmax = np.argmax(chistsm, axis=0)
		chistmax = np.max(chistsm, axis=0)
		chistmax[chistargmax < 400] = chistsm[400, chistargmax < 400]
		chistargmax[chistargmax < 400] = 400
		chistmax[chistargmax > 65500] = chistsm[65500, chistargmax > 65500]
		chistargmax[chistargmax > 65500] = 65500
		
		
		chistsky = chistmax * skylevelfactor
		
		chistskydn = np.zeros(c.shape[2])
		for i in range(c.shape[2]):
			chistskydn[i] = np.where(chistsm[:, i] > chistsky[i])[0][-1]
		
		pskylevelfactor = skylevelfactor * 100.0
		
		chistskysub1 = chistsky - zerosky
		
		cfscale   = 65535.0 / (65535.0 - chistskysub1)
		
		rgbskysub1 = chistskydn  - zerosky
		
		c = (c - rgbskysub1) * (65535.0 / (65535.0 - rgbskysub1))
		c[c < 0] = 0
		
	return c


def root_stretch(im, args):
	x = 1.0 / rootpower
	
	rootpower = args.rootpower
	rootpower2 = args.rootpower2
	rootiter = args.rootiter
	
	c = np.array(im).astype(np.float64)
	
	for irootiter in range(0, rootiter):
		print('irootiter:', irootiter)
		
		if irootiter == 1:
			x = 1.0 / rootpower2
		
		b = c + 1.0
		b = b / 65536.0
		b = 65535.0 * (b ** x)
		
		bmin = np.min(b)
		bminz = bmin - 4096.0
		if (bminz < 0.0) bminz = 0.0
		
		b = (b - bminz)
		b = b / (65535.0 - bminz)
		c = 65535.0 * b           # scale from a 0 to 1.0 range
		# c = float(c)
		
		ispassmax = 2
		if rootpower > 60.0:
			ispassmax = 3
		else:
			ispassmax = 2
		
		c = subtract_sky(c, ispassmax, args)


# sc = (xfactor / (1 + exp(-1 * ((float(c)/65535.0 - xoffset) * xfactor) ))-(1- xoffset))/scurvemax
def s_curve(im, nscurve):
	xfactors = [5.0, 3.0, 5.0, 3.0, 5.0]
	xoffsets = [0.42, 0.22, 0.42, 0.22, 0.42]
	
	c = np.array(im).astype(np.float64)
	
	for i in range(nscurve):
		if i >= len(xfactors):
			xfactor = xfactors[-1]
			xoffset = xoffsets[-1]
		else:
			xfactor = xfactors[i]
			xoffset = xoffsets[i]
	
		scurvemin =  (xfactor / (1.0 + exp(-1.0 * ((0.0/65535.0 - xoffset) * xfactor) ))-(1.0- xoffset))
		scurvemax =  (xfactor / (1.0 + exp(-1.0 * ((65535.0/65535.0 - xoffset) * xfactor) ))-(1.0- xoffset))
		scurveminsc = scurvemin / scurvemax
		
		xo = (1.0 - xoffset)
		sc = c / 65535.0
		sc = sc - xoffset      # now have (float(c)/65535.0 - xoffset)
		sc = sc * xfactor
		sc = sc * (-1.0)
		sc = np.exp(sc)           # now have exp(-1.0 * ((float(c)/65535.0 - xoffset) * xfactor) )
	
		#sc = (xfactor / (1.0 + sc )-(1.0- xoffset))/scurvemax

		sc = 1.0 + sc          # now have (1.0 + exp(-1.0 * ((float(c)/65535.0 - xoffset) * xfactor) ))
		sc = xfactor / sc
		sc = (sc-xo)
		sc = sc / scurvemax

		# image range is now -0.00829863 to 1.0  when i=1

		#cbefore = c 
		# c = 65535.0 * (sc - scurveminsc) / (1.0 - scurveminsc)
		sc = sc - scurveminsc
		sc = 65535.0 * sc
		c = sc / (1.0 - scurveminsc)
		
	return c
	
	
def set_min(im, minval):
	c = np.array(im)
	zx = 0.2  # keep some of the low level, which is noise, so it looks more natural.
	
	for i in range(3):
		(a, b) = np.where(c[:, :, i] < minval[i])
		c[a, b, np.ones(len(a), 1) * i] = minval[i] + zx * c[a, b, np.ones(len(a), 1) * i]

	return c


def color_correct(orig_im, im, args):
	af = np.array(orig_im).astype(np.float64)
	c = np.array(im).astype(np.float64)
	
	afs = af - args.zerosky
	afs[afs < 10] = 10
	
	
	grratio = (afs[:,:,1] / afs[:,:,0]) / (c[:,:,1] / c[:,:,0])  # green / red ratio
	brratio = (afs[:,:,2] / afs[:,:,0]) / (c[:,:,2] / c[:,:,0])  # blue  / red ratio

	rgratio = (afs[:,:,0] / afs[:,:,1]) / (c[:,:,0] / c[:,:,1])  # red   / green ratio
	bgratio = (afs[:,:,2] / afs[:,:,1]) / (c[:,:,2] / c[:,:,1])  # blue  / green ratio

	gbratio = (afs[:,:,1] / afs[:,:,2]) / (c[:,:,1] / c[:,:,2])  # green / blue ratio
	rbratio = (afs[:,:,0] / afs[:,:,2]) / (c[:,:,0] / c[:,:,2])  # red   / blue ratio

	zmin = 0.2
	zmax = 1.0   # note: numbers >1 desaturate.
	
	grratio[ grratio < zmin ] = zmin
	grratio[ grratio > zmax ] = zmax

	brratio[ brratio < zmin ] = zmin
	brratio[ brratio > zmax ] = zmax

	rgratio[ rgratio < zmin ] = zmin
	rgratio[ rgratio > zmax ] = zmax

	bgratio[ bgratio < zmin ] = zmin
	bgratio[ bgratio > zmax ] = zmax

	gbratio[ gbratio < zmin ] = zmin
	gbratio[ gbratio > zmax ] = zmax

	rbratio[ rbratio < zmin ] = zmin
	rbratio[ rbratio > zmax ] = zmax
	
	cavgn = np.sum(c, axis=2) / c.shape[2]    # note this is 0 to 65535 scale
	
	cavgn = cavgn / 65535.0             # note: this image is a normalized 0 to 1.0 scale. floating point
	
	cavgn[ cavgn < 0.0 ] = 0.0
	
	cavgn /= np.max(cavgn)
	
	cavgn = cavgn ** 0.2
	
	cavgn = (cavgn +0.3) / (1.0 + 0.3)  # prevent low level from copmpletely being lost

	cfactor = 1.2
	
	cfe = cfactor * args.colorenhance * cavgn
	
	grratio = 1.0 + (cfe * (grratio - 1.0))
	brratio = 1.0 + (cfe * (brratio - 1.0))
	rgratio = 1.0 + (cfe * (rgratio - 1.0))
	bgratio = 1.0 + (cfe * (bgratio - 1.0))
	gbratio = 1.0 + (cfe * (gbratio - 1.0))
	rbratio = 1.0 + (cfe * (rbratio - 1.0))
	
	grratio = 1.0 + (cfe * (grratio - 1.0))
	brratio = 1.0 + (cfe * (brratio - 1.0))
	rgratio = 1.0 + (cfe * (rgratio - 1.0))
	bgratio = 1.0 + (cfe * (bgratio - 1.0))
	gbratio = 1.0 + (cfe * (gbratio - 1.0))
	rbratio = 1.0 + (cfe * (rbratio - 1.0))
	
	c2gr = c[:,:,2] * grratio  # green adjusted
	c3br = c[:,:,3] * brratio  # blue adjusted

	c1rg = c[:,:,1] * rgratio  # red adjusted
	c3bg = c[:,:,3] * bgratio  # blue adjusted

	c1rb = c[:,:,1] * rbratio  # red adjusted
	c2gb = c[:,:,2] * gbratio  # green adjusted
	
	max_chan_idx = np.argmax(c, axis=2)
	
	(a, b) = np.where(max_chan_idx == 0)
	c[a,b,np.ones(len(a),1)*1] = c2gr[a,b]  # green adjusted
	c[a,b,np.ones(len(a),1)*2] = c3br[a,b]  # blue adjusted
	
	(a, b) = np.where(max_chan_idx == 1)
	c[a,b,np.ones(len(a),1)*0] = c1rg[a,b]  # green adjusted
	c[a,b,np.ones(len(a),1)*2] = c3bg[a,b]  # blue adjusted
	
	(a, b) = np.where(max_chan_idx == 2)
	c[a,b,np.ones(len(a),1)*0] = c1rb[a,b]  # green adjusted
	c[a,b,np.ones(len(a),1)*1] = c2gb[a,b]  # blue adjusted
	
	return c

	
def main():
	parser = create_parser()
	args = parser.parse_args()
	im = Image.open(args.imfile)
	print('im.mode:', im.mode, 'im.size:', im.size, 'im.nframes:', im.n_frames)
	# n_chan = im.n_frames
	im.seek(0)
	im_0 = np.array(im)
	im.seek(1)
	im_1 = np.array(im)
	im.seek(2)
	im_2 = np.array(im)
	im = np.zeros((im_0.shape[0], im_0.shape[1], 3), dtype=im_0.dtype)
	im[:, :, 0] = im_0
	im[:, :, 1] = im_1
	im[:, :, 2] = im_2
	print('im.shape:', im.shape, 'im.dtype:', im.dtype)
	im_flat = np.reshape(im, (im.shape[0] * im.shape[1], im.shape[2]))
	mom = np.vstack((np.min(im_flat, axis=0), np.max(im_flat, axis=0), np.mean(im_flat, axis=0))).T
	print('mom:', mom)
	ahist = np.zeros((3, 65536), dtype=np.float64)
	ahist[0, :] = np.histogram(im[:,:,0], np.arange(0, 65537))[0]
	ahist[1, :] = np.histogram(im[:,:,1], np.arange(0, 65537))[0]
	ahist[2, :] = np.histogram(im[:,:,2], np.arange(0, 65537))[0]
	# print('len(ahist):', len(ahistred))
	# print(H[1].tolist())
	npix = im.shape[0] * im.shape[1]
	npixm = int(npix * args.pcntclip/100.0)
	print('npixm:', npixm)
	
	find_clip_lvls(ahist, npixm)
	
	c = subtract_sky(im, 2, args)
	Image.fromarray(c[:,:,0]).save('find_sky_level_r.tif')
	Image.fromarray(c[:,:,1]).save('find_sky_level_g.tif')
	Image.fromarray(c[:,:,2]).save('find_sky_level_b.tif')
	
	c = root_stretch(c, args)
	c = s_curve(c, args.scurve)
	if args.scurve > 0:
		c = subtract_sky(im, 2, args)
	if args.setmin is not None:
		c = set_min(c, args.setmin)
	if args.colorcorrect:
		c = color_correct(c, args)
	if args.setmin is not None:
		c = set_min(c, args.setmin)
	c = subtract_sky(im, 1, args)
	
	
	
if __name__ == '__main__':
	main()
