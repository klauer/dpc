#!/usr/bin/env python
'''
Created on May 23, 2013, last modified on June 19, 2013
@author: Cheng Chang (cheng.chang.ece@gmail.com)
         Computer Science Group, Computational Science Center
         Brookhaven National Laboratory

This code is for Differential Phase Contrast (DPC) imaging based on Fourier-shift fitting
implementation.

Reference: Yan, H. et al. Quantitative x-ray phase imaging at the nanoscale by multilayer
           Laue lenses. Sci. Rep. 3, 1307; DOI:10.1038/srep01307 (2013).

Test data is available at:
https://docs.google.com/file/d/0B3v6W1bQwN_AdjZwWmE3WTNqVnc/edit?usp=sharing
'''
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL

from scipy.misc import imsave
from scipy.optimize import minimize
import time
import zipfile
import cStringIO as StringIO
import load_timepix

rss_cache = {}

rss_iters = 0


def get_beta(xdata):
    length = len(xdata)
    try:
        beta = rss_cache[length]
    except:
        #beta = 1j * (np.arange(length) + 1 - (np.floor(length / 2.0) + 1))
        beta = 1j * (np.arange(length) - np.floor(length / 2.0))
        rss_cache[length] = beta

    return beta


def rss(v, xdata, ydata, beta):
    '''Function to be minimized in the Nelder Mead algorithm'''
    fitted_curve = xdata * v[0] * np.exp(v[1] * beta)
    return np.sum(np.abs(ydata - fitted_curve) ** 2)


def pil_load(fn):
    im = PIL.Image.open(fn)

    def toarray(im, dtype=np.uint8):
        x_str = im.tostring('raw', im.mode)
        return np.fromstring(x_str, dtype)

    assert(im.mode.startswith('I;16'))
    if im.mode.endswith('B'):
        x = toarray(im, '>u2')
    else:
        x = toarray(im, '<u2')

    x.shape = im.size[1], im.size[0]
    return x.astype('=u2')


def load_file(fn, roi=None, bad_pixels=[], zip_file=None):
    """
    Load an image file
    """
    if os.path.exists(fn):
        im = load_timepix.load(fn)

    elif zip_file is not None:
        raise NotImplementedError

        # loading from a zip file is just about as fast (when not running in
        # parallel)
        f = zip_file.open(fn)
        stream = StringIO.StringIO()
        stream.write(f.read())
        f.close()

        stream.seek(0)
        im = plt.imread(stream, format='tif')
    else:
        raise Exception('File not found: %s' % fn)

    if bad_pixels is not None:
        for x, y in bad_pixels:
            im[x, y] = 0

    if roi is not None:
        x1, y1, x2, y2 = roi
        im = im[x1:x2 + 1, y1:y2 + 1]

    xline = np.sum(im, axis=1)
    yline = np.sum(im, axis=0)

    fx = np.fft.fftshift(np.fft.ifft(xline))
    fy = np.fft.fftshift(np.fft.ifft(yline))
    return im, fx, fy


def xj_test(filename, i, j, roi=None, bad_pixels=[], **kwargs):
    try:
        im, fx, fy = load_file(filename, zip_file=zip_file, roi=roi,
                               bad_pixels=bad_pixels)
    except Exception as ex:
        print('Failed to load file %s: %s' % (filename, ex))
        return 0.0, 0.0, 0.0

    wx, wy = im.shape
    gx = np.sum(im[:wx / 2, :]) - np.sum(im[wx / 2:, :])
    gy = np.sum(im[:, :wy / 2]) - np.sum(im[:, wy / 2:])
    return 0, gx, gy


def run_dpc(filename, i, j, ref_fx=None, ref_fy=None,
            start_point=[1, 0],
            pixel_size=55, focus_to_det=1.46, dx=0.1, dy=0.1,
            energy=19.5, zip_file=None, roi=None, bad_pixels=[],
            max_iters=1000,
            solver='Nelder-Mead',
            invers = False):
    """
    All units in micron

    pixel_size
    focus_to_det: focus to detector distance
    dx: scan step size x
    dy: scan step size y
    energy: in keV
    """
    try:
        img, fx, fy = load_file(filename, zip_file=zip_file, roi=roi,
                                bad_pixels=bad_pixels)
    except Exception as ex:
        print('Failed to load file %s: %s' % (filename, ex))
        return 0.0, 0.0, 0.0

    #vx = fmin(rss, start_point, args=(ref_fx, fx, get_beta(ref_fx)),
    #          maxiter=max_iters, maxfun=max_iters, disp=0)
    res = minimize(rss, start_point, args=(ref_fx, fx, get_beta(ref_fx)),
                   method=solver, tol=1e-4,
                   options=dict(maxiter=max_iters))

    vx = res.x
    a = vx[0]
    if invers:
        gx = -vx[1]
    else:
        gx = vx[1]

    #vy = fmin(rss, start_point, args=(ref_fy, fy, get_beta(ref_fy)),
    #          maxiter=max_iters, maxfun=max_iters, disp=0)
    res = minimize(rss, start_point, args=(ref_fy, fy, get_beta(ref_fy)),
                   method=solver, tol=1e-6,
                   options=dict(maxiter=max_iters))

    vy = res.x
    gy = vy[1]

    #print(i, j, vx[0], vx[1], vy[1])
    return a, gx, gy


def recon(gx, gy, dx=0.1, dy=0.1, pad=1, w=1.):
    """ 
    Reconstruct the final phase image 
    Parameters
    ----------
    gx : 2-D numpy array
        phase gradient along x direction
    
    gy : 2-D numpy array
        phase gradient along y direction
    
    dx : float
        scanning step size in x direction (in micro-meter)
        
    dy : float
        scanning step size in y direction (in micro-meter)
    
    pad : float
        padding parameter
        default value, pad = 1 --> no padding
                    p p p
        pad = 3 --> p v p
                    p p p
                    
    w : float
        weighting parameter for the phase gradient along x and y direction when
        constructing the final phase image
        
    Returns
    ----------
    phi : 2-D numpy array
        final phase image
        
    References
    ----------
    [1] Yan, Hanfei, Yong S. Chu, Jorg Maser, Evgeny Nazaretski, Jungdae Kim,
    Hyon Chol Kang, Jeffrey J. Lombardo, and Wilson KS Chiu, "Quantitative
    x-ray phase imaging at the nanoscale by multilayer Laue lenses," Scientific 
    reports 3 (2013).
        
    """
    
    rows, cols = gx.shape

    gx_padding = np.zeros((pad * rows, pad * cols), dtype='d')
    gy_padding = np.zeros((pad * rows, pad * cols), dtype='d')
    
    gx_padding[(pad // 2) * rows : (pad // 2 + 1) * rows,
               (pad // 2) * cols : (pad // 2 + 1) * cols] = gx
    gy_padding[(pad // 2) * rows : (pad // 2 + 1) * rows, 
               (pad // 2) * cols : (pad // 2 + 1) * cols] = gy
    
    tx = np.fft.fftshift(np.fft.fft2(gx_padding))
    ty = np.fft.fftshift(np.fft.fft2(gy_padding))
    
    c = np.zeros((pad * rows, pad * cols), dtype=complex)
    
    mid_col = pad * cols // 2.0 + 1
    mid_row = pad * rows // 2.0 + 1

    ax = 2 * np.pi * (np.arange(pad * cols) + 1 - mid_col) / (pad * cols * dx)
    ay = 2 * np.pi * (np.arange(pad * rows) + 1 - mid_row) / (pad * rows * dy)

    kappax, kappay = np.meshgrid(ax, ay)

    c = -1j * (kappax * tx + w * kappay * ty)

    c = np.ma.masked_values(c, 0)
    c /= (kappax**2 + w * kappay**2)
    c = np.ma.filled(c, 0)

    c = np.fft.ifftshift(c)
    phi_padding = np.fft.ifft2(c)
    phi_padding = -phi_padding.real
    
    phi = phi_padding[(pad // 2) * rows : (pad // 2 + 1) * rows,
                      (pad // 2) * cols : (pad // 2 + 1) * cols]
    
    return phi
    
    
def main(file_format='SOFC/SOFC_%05d.tif',
         dx=0.1, dy=0.1,
         ref_image=1,
         zip_file=None,
         rows=121, cols=121,
         start_point=[1, 0],
         pixel_size=55,
         focus_to_det=1.46e6,
         energy=19.5,
         pool=None,
         first_image=1,
         x1=None, x2=None,
         y1=None, y2=None,
         bad_pixels=[],
         solver='Nelder-Mead',
         display_fcn=None,
         invers = False):

    print('DPC')
    print('---')
    print('\tFile format: %s' % file_format)
    print('\tdx: %s' % dx)
    print('\tdy: %s' % dy)
    print('\trows: %s' % rows)
    print('\tcols: %s' % cols)
    print('\tstart point: %s' % start_point)
    print('\tpixel size: %s' % pixel_size)
    print('\tfocus to det: %s' % (focus_to_det / 1e6))
    print('\tenergy: %s' % energy)
    print('\tfirst image: %s' % first_image)
    print('\treference image: %s' % ref_image)
    print('\tsolver: %s' % solver)
    print('\tROI: (%s, %s)-(%s, %s)' % (x1, y1, x2, y2))

    t0 = time.time()

    roi = None
    if x1 is not None and x2 is not None:
        if y1 is not None and y2 is not None:
            roi = (x1, y1, x2, y2)

    # read the reference image: only one reference image
    reference, ref_fx, ref_fy = load_file(file_format % ref_image,
                                          zip_file=zip_file, roi=roi,
                                          bad_pixels=bad_pixels)

    a = np.zeros((rows, cols), dtype='d')
    gx = np.zeros((rows, cols), dtype='d')
    gy = np.zeros((rows, cols), dtype='d')

    dpc_settings = dict(start_point=start_point,
                        pixel_size=pixel_size,
                        focus_to_det=focus_to_det,
                        dx=dx,
                        dy=dy,
                        energy=energy,
                        zip_file=zip_file,
                        ref_fx=ref_fx,
                        ref_fy=ref_fy,
                        roi=roi,
                        bad_pixels=bad_pixels,
                        solver=solver,
                        invers=invers
                        )

    def get_filename(i, j):
        frame_num = first_image + i * cols + j

        # scan 1   9669
        #          12261 images
        #        = 21930
        # scan 2   21950
        #if frame_num >= 21930:
        #    frame_num += 20
        return file_format % frame_num

    # Wavelength in micron
    lambda_ = 12.4e-4 / energy
    if pool is None:
        for i in range(rows):
            trow = time.time()
            print('Row %d' % i, end='')
            rss_iters = 0
            for j in range(cols):
                _a, _gx, _gy = run_dpc(get_filename(i, j), i, j,
                                       **dpc_settings)
                a[i, j] = _a
                gx[i, j] = _gx
                gy[i, j] = _gy

            row_elapsed = (1000 * (time.time() - trow))
            print(' elapsed %.3fms' % row_elapsed, end=' ')
            print(' (per frame %.3fms, rss iters %d)' % (row_elapsed / cols, rss_iters))

    else:
        args = [(get_filename(i, j), i, j)
                for i in range(rows)
                for j in range(cols)
                ]

        _t0 = time.time()
        try:
            if 1:
                fcn = run_dpc
            else:
                fcn = xj_test

            if display_fcn is not None:
                np.random.shuffle(args)

            results = [pool.apply_async(fcn, arg, kwds=dpc_settings)
                       for arg in args]

            if display_fcn is not None:
                total_results = len(results)
                k = 0
                while k < total_results:
                    k = 0
                    for arg, result in zip(args, results):
                        if result.ready():
                            _a, _gx, _gy = result.get()
                            fn, i, j = arg

                            a[i, j] = _a
                            gx[i, j] = _gx
                            gy[i, j] = _gy
                            k += 1

                    try:
                        gx *= len(ref_fx) * pixel_size / (lambda_ * focus_to_det * 1e6)
                        gy *= len(ref_fy) * pixel_size / (lambda_ * focus_to_det * 1e6)
                        display_fcn(a, gx, gy, None)
                    except Exception as ex:
                        print('Failed to update display: (%s) %s' % (ex.__class__.__name__, ex))

                    time.sleep(1.0)

            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print('Cancelled')
            return

        for arg, result in zip(args, results):
            fn, i, j = arg
            _a, _gx, _gy = result.get()
            a[i, j] = _a
            gx[i, j] = _gx
            gy[i, j] = _gy
            k += 1

        _t1 = time.time()
        elapsed = _t1 - _t0
        print('Multiprocess elapsed=%.3f frames=%d (per frame %.3fms)' % (elapsed, rows * cols,
                                                                          1000 * elapsed / (rows * cols)))
    
    gx *= len(ref_fx) * pixel_size / (lambda_ * focus_to_det * 1e6)
    gy *= len(ref_fy) * pixel_size / (lambda_ * focus_to_det * 1e6)
    
    dim = len(np.squeeze(gx).shape)
    
    if dim is not 1:
        imsave('a.jpg', a)
        np.savetxt('a.txt', a)
        imsave('gx.jpg', gx)
        np.savetxt('gx.txt', gx)
        imsave('gy.jpg', gy)
        np.savetxt('gy.txt', gy)

        #-------------reconstruct the final phase image using gx and gy--------------------#
        phi = recon(gx, gy, dx, dy)
        imsave('phi.jpg', phi)
        np.savetxt('phi.txt', phi)

        t1 = time.time()
        print('Elapsed', t1 - t0)

        return a, gx, gy, phi
        
    else:
        """
        #plt.hold(False)
        plt.plot(np.squeeze(a), '-*')
        plt.savefig('a.jpg')
        np.savetxt('a.txt', a)
        
        plt.plot(np.squeeze(gx), '-*')
        plt.savefig('gx.jpg')
        np.savetxt('gx.txt', gx)
        
        plt.plot(np.squeeze(gy), '-*')
        plt.savefig('gy.jpg')
        np.savetxt('gy.txt', gy)
        
        t1 = time.time()
        print('Elapsed', t1 - t0)
        """
        
        phi = None
        return a, gx, gy, phi
    
    #plt.imshow(phi, cmap=cm.Greys_r)
    #plt.show()

if __name__ == '__main__':
    zip_file = None  # zipfile.ZipFile('SOFC.zip')
    main(zip_file=zip_file, processes=0, rows=121, cols=121)
