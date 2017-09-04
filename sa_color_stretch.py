#
# Copyright (C) Stanislaw Adaszewski, 2017
#

from argparse import ArgumentParser
from PIL import Image
import numpy as np
import os


def rgb_to_hsv(c):
	# c = c.astype(np.float64) / 65535 # to 0 -> 1
	r = c[:, :, 0]
	g = c[:, :, 1]
	b = c[:, :, 2]
	amax = np.argmax(c, axis=2)
	amin = np.argmin(c, axis=2)
	[yy, xx] = np.mgrid[0:c.shape[0], 0:c.shape[1]]
	yy = np.ravel(yy)
	xx = np.ravel(xx)
	cmax = np.reshape(c[yy, xx, np.ravel(amax)], c.shape[0:2])
	cmin = np.reshape(c[yy, xx, np.ravel(amin)], c.shape[0:2])
	delta = cmax - cmin
	hue = np.zeros(cmax.shape)
	sat = np.zeros(cmax.shape)
	hue[amax == 0] = 60 * ((g[amax == 0] - b[amax == 0]) / delta[amax == 0])
	hue[amax == 1] = 60 * ((b[amax == 1] - r[amax == 1]) / delta[amax == 1] + 2)
	hue[amax == 2] = 60 * ((r[amax == 2] - g[amax == 2]) / delta[amax == 2] + 4)
	hue[hue < 0] += 360
	sat[cmax == 0] = 0
	sat[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
	hsv = np.zeros((c.shape[0], c.shape[1], 3))
	hsv[:, :, 0] = hue
	hsv[:, :, 1] = sat
	hsv[:, :, 2] = cmax
	return hsv
	
	
def hsv_to_rgb(c):
	h = c[:, :, 0]
	s = c[:, :, 1]
	v = c[:, :, 2]

	hh = h / 60
	i = np.floor(hh)
	ff = hh - i
	p = v * (1 - s)
	q = v * (1 - (s * ff))
	t = v * (1 - (s * (1 - ff)))
	
	r = np.zeros(c.shape[0:2])
	g = np.zeros(c.shape[0:2])
	b = np.zeros(c.shape[0:2])
	
	mask = (i == 0)
	r[mask] = v[mask]
	g[mask] = t[mask]
	b[mask] = p[mask]

	mask = (i == 1)
	r[mask] = q[mask]
	g[mask] = v[mask]
	b[mask] = p[mask]

	mask = (i == 2)
	r[mask] = p[mask]
	g[mask] = v[mask]
	b[mask] = t[mask]

	mask = (i == 3)
	r[mask] = p[mask]
	g[mask] = q[mask]
	b[mask] = v[mask]

	mask = (i == 4)
	r[mask] = t[mask]
	g[mask] = p[mask]
	b[mask] = v[mask]

	mask = (i == 5)
	r[mask] = v[mask]
	g[mask] = p[mask]
	b[mask] = q[mask]
	
	rgb = np.zeros((c.shape[0], c.shape[1], 3))
	rgb[:, :, 0] = r
	rgb[:, :, 1] = g
	rgb[:, :, 2] = b
	return rgb


def create_parser():
	parser = ArgumentParser()
	parser.add_argument('imfile', type=str)
	parser.add_argument('--rootpower', type=float, default=6.0)
	parser.add_argument('--rootiter', type=int, default=1)
	parser.add_argument('--rootpower2', type=float, default=1.0)
	parser.add_argument('--pcntclip', type=float, default=0.005)
	parser.add_argument('--nocolorcorrect', action='store_true')
	parser.add_argument('--colorenhance', type=float, default=1.0)
	parser.add_argument('--tonecurve', action='store_true')
	parser.add_argument('--cumstretch', action='store_true')
	parser.add_argument('--skylevelfactor', type=float, default=0.06)
	# parser.add_argument('--rgbskylevel', type=float, default=1024.0)
	parser.add_argument('--rgbskylevelrs', type=float, default=4096.0)
	parser.add_argument('--zerosky', type=float, default=4096.0, nargs=1)
	# parser.add_argument('--zeroskyred', type=float, default=4096.0)
	# parser.add_argument('--zeroskygreen', type=float, default=4096.0)
	# parser.add_argument('--zeroskyblue', type=float, default=4096.0)
	parser.add_argument('--scurve', type=int, default=0)
	parser.add_argument('--setmin', type=float, nargs=1)
	# parser.add_argument('--setminr', type=float, default=0.0)
	# parser.add_argument('--setming', type=float, default=0.0)
	# parser.add_argument('--setminb', type=float, default=0.0)
	return parser
	
	
def tone_curve(im):
	af = np.array(im)

	#		 af*b*(1/d)^((af/c)^0.4)
	# print("applying tone curve...")
	b=12.0
	c=65535.0
	d=12.0
	af=af*b*((1.0/d)**((af/c)**0.4))

        # a = format(af, format=int)	# 32-bit integer

	# amomred = moment(a[,,1])
	# amomgreen = moment(a[,,2])
	# amomblue = moment(a[,,3])

	# printf ("input image after application of tone curve:\n")
	# printf("	   RED:	min=%d	max=%d   mean=%d\n",  amomred[1,,],   amomred[2,,],   amomred[3,,])
	# printf("	   GREEN:  min=%d	max=%d   mean=%d\n",  amomgreen[1,,], amomgreen[2,,], amomgreen[3,,])
	# printf("	   BLUE:   min=%d	max=%d   mean=%d\n",  amomblue[1,,],  amomblue[2,,],  amomblue[3,,])
	return af
	
	
# def find_clip_lvls(ahist, npixm):
	# alow = np.zeros(ahist.shape[0])
	
	# ahistcum = np.cumsum(ahist, axis=1)
	# for i in range(0, ahist.shape[0]):
		# idx = np.where(ahistcum[i, :]>=npixm)[0]
		# print('clip level for chan %d: %f' % (i, idx[0]))
		# if idx[0] > 39900:
			# print('clipping to 39900')
			# alow[i] = 39900
		# else:
			# alow[i] = idx[0]
	
	# return alow
	
	
# def moving_avg(h, ism):
	# res = np.zeros(h.shape, dtype=h.dtype)
	# for ih in range(0, 65536):
		# ilow = ih - ism
		# if ilow < 0: ilow = 0
		# ihigh = ih + ism
		# if ihigh > 65535: ihigh = 65535
		# res[ih] = np.mean(h[ilow:ihigh+1])
	# return res


def subtract_sky(im, npass, args):

	skylevelfactor = args.skylevelfactor
	zerosky = args.zerosky
	
	print('skylevelfactor:', skylevelfactor)
	print('zerosky:', zerosky)
	print('npass:', npass)
	print('im.shape:', im.shape)

	c = np.array(im)
	
	for pass_ in range(0, npass):
		print('pass_:', pass_)
		chist = np.zeros((65536, c.shape[2]), dtype=np.float64)
		for i in range(0, c.shape[2]):
			chist[:, i] = np.histogram(c[:, :, i], np.arange(0, 65537))[0]
			
		chistsm = np.zeros(chist.shape) # smoothed
		ism = 300
		
		for i in range(c.shape[2]):
			# chistsm[:, i] = moving_avg(chist[:, i], ism)
			chistsm[:, i] = np.convolve(chist[:, i], np.ones(ism)/ism, mode='same')
			
		chistargmax = np.argmax(chistsm, axis=0)
		chistmax = np.max(chistsm, axis=0)
		# print('#1 chistargmax:', chistargmax, 'chistmax:', chistmax)
		chistmax[chistargmax < 400] = chistsm[400, chistargmax < 400]
		# print('#2 chistargmax:', chistargmax, 'chistmax:', chistmax)
		chistargmax[chistargmax < 400] = 400
		chistmax[chistargmax > 65500] = chistsm[65500, chistargmax > 65500]
		chistargmax[chistargmax > 65500] = 65500
		
		# if np.any(chistargmax == 0):
			# print('Cannot detect sky level. Fail!')
			# raise ArgumentException('Cannot detect sky level')
		
		chistsky = chistmax * skylevelfactor
		
		chistskydn = np.zeros(c.shape[2])
		for i in range(c.shape[2]):
			chistskydn[i] = np.where(chistsm[:, i] > chistsky[i])[0][-1]
			
		if np.any(chistskydn == 0):
			print('Cannot detect sky level. Fail!')
			raise ArgumentException('Cannot detect sky level')
			
		print('chistskydn:', chistskydn)
		
		pskylevelfactor = skylevelfactor * 100.0
		
		chistskysub1 = chistsky - zerosky
		
		cfscale   = 65535.0 / (65535.0 - chistskysub1)
		
		rgbskysub1 = chistskydn  - zerosky
		
		c = (c - rgbskysub1) * (65535.0 / (65535.0 - rgbskysub1))
		c[c < 0] = 0
		
	return c


def root_stretch(im, args):
	rootpower = args.rootpower
	rootpower2 = args.rootpower2 if args.rootpower2 > 1.0 else rootpower
	rootiter = args.rootiter
	
	print('rootpower:', rootpower, 'rootpower2:', rootpower2)
	print('rootiter:', rootiter)
	
	x = 1.0 / rootpower
	print('x:', x)
	
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
		if bminz < 0.0:
			bminz = 0.0
		
		b = (b - bminz)
		b = b / (65535.0 - bminz)
		c = 65535.0 * b		   # scale from a 0 to 1.0 range
		# c = float(c)
		
		ispassmax = 2
		if rootpower > 60.0:
			ispassmax = 3
		else:
			ispassmax = 2
		
		# args_1 = lambda: 0
		# args_1.skylevelfactor = 0.06
		# args_1.zerosky = args.zerosky
		# args_1.rootpower2 = rootpower2
		# c = subtract_sky(c, ispassmax, args)
		
	return c


# sc = (xfactor / (1 + exp(-1 * ((float(c)/65535.0 - xoffset) * xfactor) ))-(1- xoffset))/scurvemax
def s_curve(im, nscurve):
	print('nscurve:', nscurve)

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
	
		scurvemin =  (xfactor / (1.0 + np.exp(-1.0 * ((0.0/65535.0 - xoffset) * xfactor) ))-(1.0- xoffset))
		scurvemax =  (xfactor / (1.0 + np.exp(-1.0 * ((65535.0/65535.0 - xoffset) * xfactor) ))-(1.0- xoffset))
		scurveminsc = scurvemin / scurvemax
		
		xo = (1.0 - xoffset)
		sc = c / 65535.0
		sc = sc - xoffset	  # now have (float(c)/65535.0 - xoffset)
		sc = sc * xfactor
		sc = sc * (-1.0)
		sc = np.exp(sc)		   # now have exp(-1.0 * ((float(c)/65535.0 - xoffset) * xfactor) )
	
		#sc = (xfactor / (1.0 + sc )-(1.0- xoffset))/scurvemax

		sc = 1.0 + sc		  # now have (1.0 + exp(-1.0 * ((float(c)/65535.0 - xoffset) * xfactor) ))
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
	
	
def set_min(orig_im, im, minval):
	af = np.array(im)
	c = np.array(im)
	zx = 0.2  # keep some of the low level, which is noise, so it looks more natural.
	
	for i in range(c.shape[2]):
		(a, b) = np.where(c[:, :, i] < minval[i])
		for k in range(len(a)):
			c[a[k], b[k], i] = minval[i] + zx * af[a[k], b[k], i]

	return c


# def color_correct(orig_im, im, args):
	# af = np.array(orig_im).astype(np.float64)
	# c = np.array(im).astype(np.float64)
	
	# print('af.shape:', af.shape)
	# print('c.shape:', c.shape)
	
	# afs = af - args.zerosky
	# afs[afs < 10] = 10
	
	
	# grratio = (afs[:,:,1] / afs[:,:,0]) / (c[:,:,1] / c[:,:,0])  # green / red ratio
	# brratio = (afs[:,:,2] / afs[:,:,0]) / (c[:,:,2] / c[:,:,0])  # blue  / red ratio

	# rgratio = (afs[:,:,0] / afs[:,:,1]) / (c[:,:,0] / c[:,:,1])  # red   / green ratio
	# bgratio = (afs[:,:,2] / afs[:,:,1]) / (c[:,:,2] / c[:,:,1])  # blue  / green ratio

	# gbratio = (afs[:,:,1] / afs[:,:,2]) / (c[:,:,1] / c[:,:,2])  # green / blue ratio
	# rbratio = (afs[:,:,0] / afs[:,:,2]) / (c[:,:,0] / c[:,:,2])  # red   / blue ratio

	# zmin = 0.2
	# zmax = 1.0   # note: numbers >1 desaturate.
	
	# grratio[ grratio < zmin ] = zmin
	# grratio[ grratio > zmax ] = zmax

	# brratio[ brratio < zmin ] = zmin
	# brratio[ brratio > zmax ] = zmax

	# rgratio[ rgratio < zmin ] = zmin
	# rgratio[ rgratio > zmax ] = zmax

	# bgratio[ bgratio < zmin ] = zmin
	# bgratio[ bgratio > zmax ] = zmax

	# gbratio[ gbratio < zmin ] = zmin
	# gbratio[ gbratio > zmax ] = zmax

	# rbratio[ rbratio < zmin ] = zmin
	# rbratio[ rbratio > zmax ] = zmax
	
	# cavgn = np.sum(c, axis=2) / c.shape[2]	# note this is 0 to 65535 scale
	
	# cavgn = cavgn / 65535.0			 # note: this image is a normalized 0 to 1.0 scale. floating point
	
	# cavgn[ cavgn < 0.0 ] = 0.0
	
	# cavgn /= np.max(cavgn)
	
	# cavgn = cavgn ** 0.2
	
	# cavgn = (cavgn +0.3) / (1.0 + 0.3)  # prevent low level from copmpletely being lost

	# cfactor = 1.2
	
	# cfe = cfactor * args.colorenhance * cavgn
	
	# grratio = 1.0 + (cfe * (grratio - 1.0))
	# brratio = 1.0 + (cfe * (brratio - 1.0))
	# rgratio = 1.0 + (cfe * (rgratio - 1.0))
	# bgratio = 1.0 + (cfe * (bgratio - 1.0))
	# gbratio = 1.0 + (cfe * (gbratio - 1.0))
	# rbratio = 1.0 + (cfe * (rbratio - 1.0))
	
	# c2gr = c[:,:,1] * grratio  # green adjusted
	# c3br = c[:,:,2] * brratio  # blue adjusted

	# c1rg = c[:,:,0] * rgratio  # red adjusted
	# c3bg = c[:,:,2] * bgratio  # blue adjusted

	# c1rb = c[:,:,0] * rbratio  # red adjusted
	# c2gb = c[:,:,1] * gbratio  # green adjusted
	
	# max_chan_idx = np.argmax(c, axis=2)
	
	# (a, b) = np.where(max_chan_idx == 0)
	# print('len(a):', len(a), 'len(b):', len(b))
	# for i in range(len(a)):
		# c[a[i],b[i],1] = c2gr[a[i],b[i]]  # green adjusted
		# c[a[i],b[i],2] = c3br[a[i],b[i]]  # blue adjusted
	
	# (a, b) = np.where(max_chan_idx == 1)
	# print('len(a):', len(a), 'len(b):', len(b))
	# for i in range(len(a)):
		# c[a[i],b[i],0] = c1rg[a[i],b[i]]  # green adjusted
		# c[a[i],b[i],2] = c3bg[a[i],b[i]]  # blue adjusted
	
	# (a, b) = np.where(max_chan_idx == 2)
	# print('len(a):', len(a), 'len(b):', len(b))
	# for i in range(len(a)):
		# c[a[i],b[i],0] = c1rb[a[i],b[i]]  # green adjusted
		# c[a[i],b[i],1] = c2gb[a[i],b[i]]  # blue adjusted
	
	# return c
	
	
def save_result(c, suffix, args):
	# cmin = np.min(c)
	# cmax = np.max(c)
	c = c.astype(np.float64) * 255 / 65535 # (cmax - cmin)
	
	print('Saving result...')
	(dname, fname) = os.path.split(args.imfile)
	(fname, ext) = os.path.splitext(fname)
	outname = '%s_%s.png' % (fname, suffix)
	print('outname:', outname)
	Image.fromarray(np.squeeze(c.astype(np.uint8))).save(outname)

	
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
	
	cmin = np.min(im)
	cmax = np.max(im)
	print('cmin:', cmin, 'cmax:', cmax)
	# im = 100 + (im.astype(np.float64) - cmin) * 65000 / cmax
	# im = im.astype(np.uint16)
	
	im_flat = np.ravel(im) # np.reshape(im, (im.shape[0] * im.shape[1], im.shape[2]))
	mom = np.vstack((np.min(im_flat, axis=0), np.max(im_flat, axis=0), np.mean(im_flat, axis=0))).T
	print('mom:', mom)
	# ahist = np.zeros((3, 65536), dtype=np.float64)
	# ahist[0, :] = np.histogram(im[:,:,0], np.arange(0, 65537))[0]
	# ahist[1, :] = np.histogram(im[:,:,1], np.arange(0, 65537))[0]
	# ahist[2, :] = np.histogram(im[:,:,2], np.arange(0, 65537))[0]
	# print('len(ahist):', len(ahistred))
	# print(H[1].tolist())
	npix = im.shape[0] * im.shape[1]
	npixm = int(npix * args.pcntclip/100.0)
	print('npixm:', npixm)
	
	print('Converting RGB to HSV...')
	hsv = rgb_to_hsv(im / 65535.0)
	
	c = hsv[:, :, 2] * 65535 # work on value only
	c = c[:, :, np.newaxis]
	print('Working on value only...')
	print('c.shape:', c.shape)
	
	# using c from now on
	# c = im
	
	# find_clip_lvls(ahist, npixm)
	if args.tonecurve:
		print('---> Applying tone curve...')
		c = tone_curve(im)
		save_result(c, 'tone_curve', args)
		print('c.shape:', c.shape)
	
	print('---> First sky-subtract...')
	c = subtract_sky(c, 2, args)
	# Image.fromarray(c[:,:,0]).save('find_sky_level_r.tif')
	# Image.fromarray(c[:,:,1]).save('find_sky_level_g.tif')
	# Image.fromarray(c[:,:,2]).save('find_sky_level_b.tif')
	save_result(c, '1st_sky_sub', args)
	print('c.shape:', c.shape)
	
	if args.rootiter > 0:
		print('---> Root stretching...')
		c = root_stretch(c, args)
		# c = subtract_sky(c, 1, args)
		save_result(c, 'rs', args)
		
		

		
	# if args.setmin is not None:
		# print('---> Set-min...')
		# c = set_min(c, c, args.setmin)
		
	# if not args.nocolorcorrect:
		# print('---> Color correction...')
		# c = color_correct(im, c, args)
		# save_result(c, 'cc', args)
		
	# if args.setmin is not None:
		# print('---> Second Set-min...')
		# c = set_min(c, c, args.setmin)
		
	# print('---> Final sky-subtract...')
	# c = subtract_sky(c, 1, args)
	# save_result(c, 'vfin', args)
	
	print('Recomposing HSV...')
	# hsv[:, :, 1] = s_curve(hsv[:, :, 1], args.scurve)
	print('max(c):', np.max(c))
	hsv[:, :, 2] = np.squeeze(c) / 65535
	c = hsv_to_rgb(hsv) * 65535
	
	if args.scurve > 0:
		print('---> S-Curve...')
		c = s_curve(c, args.scurve)
		save_result(c, 'sc', args)
		# print('---> Subtracting sky after S-Curve...')
		# c = subtract_sky(c, 2, args)
	
	# if args.scurve > 0:
		# print('---> S-Curve...')
		# c = s_curve(c, args.scurve)
		# print('---> Subtracting sky after S-Curve...')
		# c = subtract_sky(c, 2, args)
	
	# print('Scaling to 8-bit...')
	save_result(c, 'sa', args)
	
	print('Done.')
	
	
if __name__ == '__main__':
	main()
