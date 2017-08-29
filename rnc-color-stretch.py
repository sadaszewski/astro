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
	parser.add_argument('--rgbskylevel', type=float, default=1024.0)
	parser.add_argument('--rgbskylevelrs', type=float, default=4096.0)
	parser.add_argument('--zerosky', type=float, default=[4096.0, 4096.0, 4096.0], nargs=3)
	# parser.add_argument('--zeroskyred', type=float, default=4096.0)
	# parser.add_argument('--zeroskygreen', type=float, default=4096.0)
	# parser.add_argument('--zeroskyblue', type=float, default=4096.0)
	parser.add_argument('--scurve', action='store_true')
	parser.add_argument('--setmin', action='store_true')
	parser.add_argument('--setminr', type=float, default=0.0)
	parser.add_argument('--setming', type=float, default=0.0)
	parser.add_argument('--setminb', type=float, default=0.0)
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


def find_sky_level(im, args):

	c = np.array(im)
	
	for pass_ in range(0, 2):
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
		
		
		chistsky = chistmax * args.skylevelfactor
		
		chistskydn = np.zeros(c.shape[2])
		for i in range(c.shape[2]):
			chistskydn[i] = np.where(chistsm[:, i] > chistsky[i])[0][-1]
		
		pskylevelfactor = args.skylevelfactor * 100.0
		
		chistskysub1 = chistsky - args.zerosky
		
		cfscale   = 65535.0 / (65535.0 - chistskysub1)
		
		rgbskysub1 = chistskydn  - args.zerosky
		
		c = (c - rgbskysub1) * (65535.0 / (65535.0 - rgbskysub1))
		c[c < 0] = 0
		
	return c
		
	
	'''c = af

   for ( ispass=1; ispass<=2;  ispass++ ) {    # do two pass on finding sky level
	printf ("\n\n")
	printf ("     computing RGB histograms on input image and smoothing the histograms, pass %d\n", ispass )

	chistred   = float(histogram(c[,,1], start=0.0, size=1.0, steps=65536))
	chistgreen = float(histogram(c[,,2], start=0.0, size=1.0, steps=65536))
	chistblue  = float(histogram(c[,,3], start=0.0, size=1.0, steps=65536))

	chistredsm   =  chistred     # array for smoothed histograms
	chistgreensm =  chistgreen
	chistbluesm  =  chistblue

	ism=300  # number of chanels to smooth

	for (ih=1; ih<=65535; ih++) {   # smooth the histograms

		ilow = ih - ism
		if (ilow < 1) ilow = 1
		ihigh = ih + ism
		if (ihigh > 65535) ihigh = 65535

		csr= avg(chistred[2,ilow:ihigh,1], axis=y)
		chistredsm[2,ih,1] = csr    # smoothed histogram value at ih
						#printf ("debug: ih, snch= %d  %f\n", ih, cs[2,ih,1])

		csg = avg(chistgreen[2,ilow:ihigh,1], axis=y)
		chistgreensm[2,ih,1] = csg

		csb = avg(chistblue[2,ilow:ihigh,1], axis=y)
		chistbluesm[2,ih,1] = csb
	}
	# ok, we have the smoothed histograms for each color.  Now find the max.

	chistredsmax   = 0.0
	chistgreensmax = 0.0
	chistbluesmax  = 0.0

	chistredsmaxdn   = 0    # for histogram DN of the max
	chistgreensmaxdn = 0
	chistbluesmaxdn  = 0


	for (ih=400; ih<=65500; ih++) {   # limit test range in case clipping or saturation

		if ( chistredsm[2,ih,1]   > chistredsmax )   {
			chistredsmax   = chistredsm[2,ih,1]
			chistredsmaxdn = ih
		}
		if ( chistgreensm[2,ih,1] > chistgreensmax ) {
			chistgreensmax   = chistgreensm[2,ih,1]
			chistgreensmaxdn = ih
		}
		if ( chistbluesm[2,ih,1]  > chistbluesmax )  {
			chistbluesmax   = chistbluesm[2,ih,1]
			chistbluesmaxdn = ih
		}
	}

	if (chistredsmaxdn == 0 || chistgreensmaxdn == 0 || chistbluesmaxdn == 0) {

		printf ("     histogram max on input image not found:\n")
		printf ("          channels: red %d   green %d   blue %d\n", chistredsmaxdn, chistgreensmaxdn, chistbluesmaxdn)
		printf ("          This means that the algorithm can't work on this image\n")
		printf ("          Try increasing -rgbskyzero values\n")

		if (doplots == 1 && ispass == 1) {   # plot histogram

			if (idisplay > 0) {
				ajpg = byte(c/256.0)
				display(ajpg)
			}

			hrange = 65535
			xah = chistgreen[1,1:hrange,1]
			plot(chistredsm[2,1:hrange,1], axis=y, xaxis=xah, \
					label="histograms on root stretched image, smoothed, red", \
				chistgreensm[2,1:hrange,1], axis=y, xaxis=xah, label="green", \
				chistbluesm[2,1:hrange,1], axis=y, xaxis=xah, label="blue")

			plot("set xlabel \"Image Data DN Level\"")
			plot("set ylabel \"Number of Pixels (Linear Count)\"")
			plot("replot")

			printf ("histograms on root stretched image, smoothed\n")
			if ( iopversion < 2 ) {  # linux, macs
				printf ("  Press return to continue\n")
				system("sh -c 'read a'")
			} else {
				system("pause")   # windows
			}
		}

		printf ("       exit (1)\n")
		exit (1)
	}

	printf ("\n")
	printf ("     Histogram peak, input image:\n")
	printf ("     image histogram:     DN      Number of pixels in peak histogram bin\n")
	printf ("                  red:   %d         %f\n", chistredsmaxdn,   chistredsmax)
	printf ("                green:   %d         %f\n", chistgreensmaxdn, chistgreensmax)
	printf ("                 blue:   %d         %f\n", chistbluesmaxdn,  chistbluesmax)

	# now find the sky level on the left side of the histogram

	chistredsky   = chistredsmax   * skylevelfactor
	chistgreensky = chistgreensmax * skylevelfactor
	chistbluesky  = chistbluesmax  * skylevelfactor

	chistredskydn   = 0   # which DN is at the skylevelfactor level.  DN = Data Number
	chistgreenskydn = 0
	chistblueskydn  = 0

	for (ih = chistredsmaxdn;  ih>=2; ih=ih-1) {   # search from max toward left minimum
							# but search for the green level in each color

		if (chistredsm[2,ih,1] >= chistgreensky && \
				chistredsm[2,ih-1,1] <= chistgreensky && chistredskydn == 0) {

			chistredskydn = ih
			#printf ("debug: chistredskydn= %d\n", chistredskydn)
		}
	}
	for (ih = chistgreensmaxdn;  ih>=2; ih=ih-1) {   # search from max toward left minimum

		if (chistgreensm[2,ih,1] >= chistgreensky && \
				chistgreensm[2,ih-1,1] <= chistgreensky && chistgreenskydn == 0) {

			chistgreenskydn = ih
			#printf ("debug: chistgreenskydn= %d\n", chistgreenskydn)
		}
	}
	for (ih = chistbluesmaxdn;  ih>=2; ih=ih-1) {   # search from max toward left minimum

		if (chistbluesm[2,ih,1] >= chistgreensky && \
				chistbluesm[2,ih-1,1] <= chistgreensky && chistblueskydn == 0) {

			chistblueskydn = ih
			#printf ("debug: chistblueskydn= %d\n", chistblueskydn)
		}
	}
	if ( chistredskydn == 0 || chistgreenskydn == 0 || chistblueskydn == 0) {

		printf ("histogram sky level %f not found:\n", skylevelfactor)
		printf ("     channels: red %d   green %d   blue %d\n", chistredskydn, chistgreenskydn, chistblueskydn)
		printf ("     Thus, not sure what to do with this image because the histogram sky level is too low, quitting\n")
		printf ("          Try increasing -rgbskyzero values\n")
		printf ("          The input data may be too low.\n")

		if (doplots == 1 && ispass == 1) {   # plot histogram

			if (idisplay > 0) {
				ajpg = byte(c/256.0)
				display(ajpg)
			}

			hrange = 65535
			xah = chistgreen[1,1:hrange,1]
			plot(chistredsm[2,1:hrange,1], axis=y, xaxis=xah, \
					label="histograms on input image, smoothed, red", \
				chistgreensm[2,1:hrange,1], axis=y, xaxis=xah, label="green", \
				chistbluesm[2,1:hrange,1], axis=y, xaxis=xah, label="blue")

			plot("set xlabel \"Image Data DN Level\"")
			plot("set ylabel \"Number of Pixels (Linear Count)\"")
			plot("replot")

			printf ("histograms on input image, smoothed\n")
			if ( iopversion < 2 ) {  # linux, macs
				printf ("  Press return to continue\n")
				system("sh -c 'read a'")
			} else {
				system("pause")   # windows
			}
		}

		printf ("  exit (1)\n")
		exit (1)
	}

	pskylevelfactor = skylevelfactor * 100.0

	printf ("\n")
	printf ("     histogram dark sky level on input image (%5.1f %% of max)\n", pskylevelfactor)
	printf ("                  DN      Number of pixels in histogram bin\n")     # need to compute cumulative hstogram
	printf ("           red:   %d         %f\n", chistredskydn,   chistredsky)
	printf ("         green:   %d         %f\n", chistgreenskydn, chistgreensky)
	printf ("          blue:   %d         %f\n", chistblueskydn,  chistbluesky)

	if (doplots == 1 && ispass == 1) {   # plot histogram

			if (idisplay > 0) {
				ajpg = byte(c/256.0)
				display(ajpg)
			}

		hrange = 65535
		xah = chistgreen[1,1:hrange,1]
		plabel = " "
		plabel = sprintf("histograms on input image, smoothed, before sky subtraction, pass %d, red", ispass)
		plot(chistredsm[2,1:hrange,1], axis=y, xaxis=xah, label=plabel, \
			chistgreensm[2,1:hrange,1], axis=y, xaxis=xah, label="green", \
			chistbluesm[2,1:hrange,1], axis=y, xaxis=xah, label="blue")

		plot("set xlabel \"Image Data DN Level\"")
		plot("set ylabel \"Number of Pixels (Linear Count)\"")
		plot("replot")

		printf ("histograms on input image, smoothed, before sky subtraction, pass %d\n", ispass)
			if ( iopversion < 2 ) {  # linux, macs
				printf ("  Press return to continue\n")
				system("sh -c 'read a'")
			} else {
				system("pause")   # windows
			}

	}


	# pre v0.891:
	#printf ("\n     Assume dark sky is neutral black, and green, %d, is the neutral level\n", chistgreenskydn)

	chistredskysub1    = chistredskydn   - zeroskyred   # subtract value to bring red   sky equal to reference zero sky level
	chistgreenskysub1  = chistgreenskydn - zeroskygreen # subtract value to bring green sky equal to reference zero sky level
	chistblueskysub1   = chistblueskydn  - zeroskyblue  # subtract value to bring blue  sky equal to reference zero sky level

	printf ("\n")
	printf ("subtract %7.1f from red   to make red   sky align with zero reference sky: %7.1f\n", chistredskysub1,   zeroskyred)
	printf ("subtract %7.1f from green to make green sky align with zero reference sky: %7.1f\n", chistgreenskysub1, zeroskygreen)
	printf ("subtract %7.1f from blue  to make blue  sky align with zero reference sky: %7.1f\n", chistblueskysub1,  zeroskyblue)

	cfscalered   = 65535.0 / (65535.0 - float(chistredskysub1))     # factor to scale so max = 65535
	cfscalegreen = 65535.0 / (65535.0 - float(chistgreenskysub1))   # factor to scale so max = 65535
	cfscaleblue  = 65535.0 / (65535.0 - float(chistblueskysub1))    # factor to scale so max = 65535

	printf ("\n")

	printf ("now set the RGB sky zero level to %7.1f  %7.1f  %7.1f DN out of 65535\n", zeroskyred, zeroskygreen, zeroskyblue)

	rgbskysub1r = float(chistredskydn   - zeroskyred)
	rgbskysub1g = float(chistgreenskydn - zeroskygreen)
	rgbskysub1b = float(chistblueskydn  - zeroskyblue)

	c[,,1] = (c[,,1] - rgbskysub1r) * (65535.0 / (65535.0 - rgbskysub1r))  # red   subtracted
	c[,,2] = (c[,,2] - rgbskysub1g) * (65535.0 / (65535.0 - rgbskysub1g))  # green subtracted
	c[,,3] = (c[,,3] - rgbskysub1b) * (65535.0 / (65535.0 - rgbskysub1b))  # blue  subtracted

	c[ where ( c < 0.0 )] = 0.0 

	cm = moment(c)
	cmr = moment(c[,,1])
	cmg = moment(c[,,2])
	cmb = moment(c[,,3])

	printf("\n")
	printf("root stretched image, power exponent= %f\n", x)
	printf("       scaled image stats, before color recovery, sky adjust pass %d:\n", ispass)
	printf("       RED:    min=%d    max=%d   mean=%d\n", int(cmr[1,,]), int(cmr[2,,]), int(cmr[3,,]))
	printf("       GREEN:  min=%d    max=%d   mean=%d\n", int(cmg[1,,]), int(cmg[2,,]), int(cmg[3,,]))
	printf("       BLUE:   min=%d    max=%d   mean=%d\n", int(cmb[1,,]), int(cmb[2,,]), int(cmb[3,,]))
	printf("\n")

	if (doplots == 1) {   # plot histogram

			if (idisplay > 0) {
				ajpg = byte(c/256.0)
				display(ajpg)
			}

		chistred   = float(histogram(c[,,1], start=0.0, size=1.0, steps=65536))
		chistgreen = float(histogram(c[,,2], start=0.0, size=1.0, steps=65536))
		chistblue  = float(histogram(c[,,3], start=0.0, size=1.0, steps=65536))

		chistredz   = chistred
		chistgreenz = chistgreen
		chistbluez  = chistblue

		for (kz=2; kz<=65535; kz++) {  # fill zeros for plor

			if (chistredz[2,kz,1]   < 0.1) chistredz[2,kz,1]   = chistredz[2,kz-1,1]
			if (chistgreenz[2,kz,1] < 0.1) chistgreenz[2,kz,1] = chistgreenz[2,kz-1,1]
			if (chistbluez[2,kz,1]  < 0.1) chistbluez[2,kz,1]  = chistbluez[2,kz-1,1]
		}

		hrange = 65535
		xah = chistgreen[1,1:hrange,1]
		plabel = sprintf("histograms on input image, sky subtraction, pass %d, red", ispass)
		plot(chistredz[2,1:hrange,1], axis=y, xaxis=xah, \
				label=plabel, \
			chistgreenz[2,1:hrange,1], axis=y, xaxis=xah, label="green", \
			chistbluez[2,1:hrange,1], axis=y, xaxis=xah, label="blue")

		plot("set xlabel \"Image Data DN Level\"")
		plot("set ylabel \"Number of Pixels (Linear Count)\"")
		plot("replot")

		printf ("histograms on input image, sky subtraction, pass %d\n", ispass)
			if ( iopversion < 2 ) {  # linux, macs
				printf ("Press return to continue\n")
				system("sh -c 'read a'")
			} else {
				system("pause")   # windows
			}

	}


   }  # end loop for multi-pass sky level'''

	
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
	c = find_sky_level(im, args)
	Image.fromarray(c[:,:,0]).save('find_sky_level_r.tif')
	Image.fromarray(c[:,:,1]).save('find_sky_level_g.tif')
	Image.fromarray(c[:,:,2]).save('find_sky_level_b.tif')
	
	
	
if __name__ == '__main__':
	main()
