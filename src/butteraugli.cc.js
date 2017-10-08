// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Jyrki Alakuijala (jyrki.alakuijala@gmail.com)
//         Dominik Homberger (dominik.homberger@gmail.com)
//
// The physical architecture of butteraugli is based on the following naming
// convention:
//   * Opsin - dynamics of the photosensitive chemicals in the retina
//             with their immediate electrical processing
//   * Xyb - hybrid opponent/trichromatic color space
//     x is roughly red-subtract-green.
//     y is yellow.
//     b is blue.
//     Xyb values are computed from Opsin mixing, not directly from rgb.
//   * Mask - for visual masking
//   * Hf - color modeling for spatially high-frequency features
//   * Lf - color modeling for spatially low-frequency features
//   * Diffmap - to cluster and build an image of error between the images
//   * Blur - to hold the smoothing code

// Purpose of kInternalGoodQualityThreshold:
// Normalize 'ok' image degradation to 1.0 across different versions of
// butteraugli.
var kInternalGoodQualityThreshold = 13.647516951250337;
var kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

//141
function DotProduct(u, v) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

//145
function ComputeKernel(sigma) {
  var m = 2.25;  // Accuracy increases when m is increased.
  var scaler = -1.0 / (2 * sigma * sigma);
  var diff = (Math.max(1, m * fabs(sigma)))|0;
  var kernel=mallocArr(2 * diff + 1,0.0);
  for (var i = -diff; i <= diff; ++i) {
    kernel[i + diff] = Math.exp(scaler * i * i);
  }
  return kernel;
}

function ConvolveBorderColumn(
    in_,
    kernel,
    weight_no_border,
    border_ratio,
    x,
    row_out,row_out_off) {
  var offset = (kernel.length / 2)|0;
  var minx = x < offset ? 0 : x - offset;
  var maxx = Math.min(in_.xsize() - 1, x + offset);
  var weight = 0.0;
  for (var j = minx; j <= maxx; ++j) {
    weight += kernel[j - x + offset];
  }
  // Interpolate linearly between the no-border scaling and border scaling.
  weight = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
  var scale = 1.0 / weight;
  for (var y = 0; y < in_.ysize(); ++y) {
    var row_in = in_.Row(y);var row_in_off = in_.Row_off(y);
    var sum = 0.0;
    for (var j = minx; j <= maxx; ++j) {
      sum += row_in[row_in_off+j] * kernel[j - x + offset];
    }
    row_out[row_out_off+y] = sum * scale;
  }
}

//184
// Computes a horizontal convolution and transposes the result.
function Convolution(in_,
                   kernel,
                   border_ratio) {
  var out=new ImageF();out._2(in_.ysize(), in_.xsize());
  var len = kernel.length;
  var offset = (kernel.length / 2)|0;
  var weight_no_border = 0.0;
  for (var j = 0; j < len; ++j) {
    weight_no_border += kernel[j];
  }
  var scale_no_border = 1.0 / weight_no_border;
  var border1 = in_.xsize() <= offset ? in_.xsize() : offset;
  var border2 = in_.xsize() - offset;
  var x = 0;
  // left border
  for (; x < border1; ++x) {
    ConvolveBorderColumn(in_, kernel, weight_no_border, border_ratio, x,
                         out.Row(x),out.Row_off(x));
  }
  // middle
  for (; x < border2; ++x) {
    var row_out = out.Row(x);var row_out_off = out.Row_off(x);
    for (var y = 0; y < in_.ysize(); ++y) {
      var row_in = in_.Row(y);var row_in_off = in_.Row_off(y)+x - offset;
      var sum = 0.0;
      for (var j = 0; j < len; ++j) {
        sum += row_in[row_in_off+j] * kernel[j];
      }
      row_out[row_out_off+y] = sum * scale_no_border;
    }
  }
  // right border
  for (; x < in_.xsize(); ++x) {
    ConvolveBorderColumn(in_, kernel, weight_no_border, border_ratio, x,
                         out.Row(x),out.Row_off(x));
  }
  return out;
}

// A blur somewhat similar to a 2D Gaussian blur.
// See: https://en.wikipedia.org/wiki/Gaussian_blur
//225
function Blur(in_, sigma, border_ratio) {
  var kernel = ComputeKernel(sigma);
  return Convolution(Convolution(in_, kernel, border_ratio),
                     kernel, border_ratio);
}

// DoGBlur is an approximate of difference of Gaussians. We use it to
// approximate LoG (Laplacian of Gaussians).
// See: https://en.wikipedia.org/wiki/Difference_of_Gaussians
// For motivation see:
// https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Laplacian_pyramid
//236
function DoGBlur(in_, sigma, border_ratio) {
  var blur1 = Blur(in_, sigma, border_ratio);
  var blur2 = Blur(in_, sigma * 2.0, border_ratio);
  var mix = 0.5;
  var out=new ImageF();out._2(in_.xsize(), in_.ysize());
  for (var y = 0; y < in_.ysize(); ++y) {
    var row1 = blur1.Row(y);var row1_off = blur1.Row_off(y);
    var row2 = blur2.Row(y);var row2_off = blur2.Row_off(y);
    var row_out = out.Row(y);var row_out_off = out.Row_off(y);
    for (var x = 0; x < in_.xsize(); ++x) {
      row_out[row_out_off+x] = (1.0 + mix) * row1[row1_off+x] - mix * row2[row2_off+x];
    }
  }
  return out;
}

// Clamping linear interpolator.
//253
function InterpolateClampNegative(array,
                                  size, ix) {
  if (ix < 0) {
    ix = 0;
  }
  var baseix = (ix)|0;//static_cast<int>
  var res;
  if (baseix >= size - 1) {
    res = array[size - 1];
  } else {
    var mix = ix - baseix;
    var nextix = baseix + 1;
    res = array[baseix] + mix * (array[nextix] - array[baseix]);
  }
  return res;
}


//322
function SimpleGamma(v) {
  var kGamma = 0.376530479761;
  var limit = 37.614164142;
  var bright = v - limit;
  if (bright >= 0) {
    var mul = 0.0658865383731;
    v -= bright * mul;
  }
  {
    var limit2 = 72.8505938033;
    var bright2 = v - limit2;
    if (bright2 >= 0) {
      var mul = 0.01;
      v -= bright2 * mul;
    }
  }
  {
    var limit2 = 82.8505938033;
    var bright2 = v - limit2;
    if (bright2 >= 0) {
      var mul = 0.047444663566;
      v -= bright2 * mul;
    }
  }
  {
    var limit2 = 92.8505938033;
    var bright2 = v - limit2;
    if (bright2 >= 0) {
      var mul = 0.208844763252;
      v -= bright2 * mul;
    }
  }
  {
    var limit2 = 102.8505938033;
    var bright2 = v - limit2;
    if (bright2 >= 0) {
      var mul = 0.041471798711500003;
      v -= bright2 * mul;
    }
  }
  {
    var limit2 = 112.8505938033;
    var bright2 = v - limit2;
    if (bright2 >= 0) {
      var mul = 0.021471798711500003;
      v -= bright2 * mul;
    }
  }
  var offset = 0.0286580002175;
  var scale = 10.5938485402;
  var retval = scale * (offset + Math.pow(v, kGamma));
  return retval;
}

//376
function Gamma(v) {
  return SimpleGamma(v);
  //return GammaPolynomial(v);
}

//381
function OpsinDynamicsImage(rgb) {
  //PROFILER_FUNC;
/*if(0) {
  PrintStatistics("rgb", rgb);
}*/
  var xyb=mallocArrOI(3,ImageF);
  var blurred=mallocArrOI(3,ImageF);
  var kSigma = 1.2744543709;
  for (var i = 0; i < 3; ++i) {
    xyb[i] = new ImageF(); xyb[i]._2(rgb[i].xsize(), rgb[i].ysize());
    blurred[i] = Blur(rgb[i], kSigma, 0.0);
  }
  for (var y = 0; y < rgb[0].ysize(); ++y) {
    var row_r = rgb[0].Row(y);var row_r_off = rgb[0].Row_off(y);
    var row_g = rgb[1].Row(y);var row_g_off = rgb[1].Row_off(y);
    var row_b = rgb[2].Row(y);var row_b_off = rgb[2].Row_off(y);
    var row_blurred_r = blurred[0].Row(y);var row_blurred_r_off = blurred[0].Row_off(y);
    var row_blurred_g = blurred[1].Row(y);var row_blurred_g_off = blurred[1].Row_off(y);
    var row_blurred_b = blurred[2].Row(y);var row_blurred_b_off = blurred[2].Row_off(y);
    var row_out_x = xyb[0].Row(y);var row_out_x_off = xyb[0].Row_off(y);
    var row_out_y = xyb[1].Row(y);var row_out_y_off = xyb[1].Row_off(y);
    var row_out_b = xyb[2].Row(y);var row_out_b_off = xyb[2].Row_off(y);
    for (var x = 0; x < rgb[0].xsize(); ++x) {
      var sensitivity=mallocArr(3,0.0);
      {
        // Calculate sensitivity based on the smoothed image gamma derivative.
        var pre_mixed0=[], pre_mixed1=[], pre_mixed2=[];
        OpsinAbsorbance(row_blurred_r[row_blurred_r_off+x], row_blurred_g[row_blurred_g_off+x], row_blurred_b[row_blurred_b_off+x],
                        pre_mixed0, pre_mixed1, pre_mixed2);//& & &
        // TODO: use new polynomial to compute Gamma(x)/x derivative.
        sensitivity[0] = Gamma(pre_mixed0[0]) / pre_mixed0[0];
        sensitivity[1] = Gamma(pre_mixed1[0]) / pre_mixed1[0];
        sensitivity[2] = Gamma(pre_mixed2[0]) / pre_mixed2[0];
      }
      var cur_mixed0=[], cur_mixed1=[], cur_mixed2=[];
      OpsinAbsorbance(row_r[row_r_off+x], row_g[row_g_off+x], row_b[row_b_off+x],
                      cur_mixed0, cur_mixed1, cur_mixed2);//& & &
      cur_mixed0[0] *= sensitivity[0];
      cur_mixed1[0] *= sensitivity[1];
      cur_mixed2[0] *= sensitivity[2];
      RgbToXyb(cur_mixed0[0], cur_mixed1[0], cur_mixed2[0],
               row_out_x,row_out_x_off+x, row_out_y,row_out_y_off+x, row_out_b,row_out_b_off+x);//& & &
    }
  }
/*if(0) {
  PrintStatistics("xyb", xyb);
}*/
  return xyb;
}

// Make area around zero less important (remove it).
//432
function RemoveRangeAroundZero(w, x) {
  return x > w ? x - w : x < -w ? x + w : 0.0;
}

// Make area around zero more important (2x it until the limit).
//437
function AmplifyRangeAroundZero(w, x) {
  return x > w ? x + w : x < -w ? x - w : 2.0 * x;
}

//441
function ModifyRangeAroundZero(warray,
                               in_) {
  var out=[];//std::vector<ImageF>
  for (var k = 0; k < 3; ++k) {
    var plane=new ImageF();plane._2(in_[k].xsize(), in_[k].ysize());
    for (var y = 0; y < plane.ysize(); ++y) {
      var row_in = in_[k].Row(y);var row_in_off = in_[k].Row_off(y);
      var row_out = plane.Row(y);var row_out_off = plane.Row_off(y);
      if (k == 2) {
        memcpy(row_out, row_out_off, row_in, row_in_off, plane.xsize());// * sizeof(row_out[0])
      } else if (warray[k] >= 0) {
        var w = warray[k];
        for (var x = 0; x < plane.xsize(); ++x) {
          row_out[row_out_off+x] = RemoveRangeAroundZero(w, row_in[row_in_off+x]);
        }
      } else {
        var w = -warray[k];
        for (var x = 0; x < plane.xsize(); ++x) {
          row_out[row_out_off+x] = AmplifyRangeAroundZero(w, row_in[row_in_off+x]);
        }
      }
    }
    out.push(plane);//emplace_back std::move()
  }
  return out;
}

// XybLowFreqToVals converts from low-frequency XYB space to the 'vals' space.
// Vals space can be converted to L2-norm space (Euclidean and normalized)
// through visual masking.
//472
function XybLowFreqToVals(x, y, b_arg,
                          valx,
                          valy,
                          valb) {
  var xmuli = 5.63685258788;
  var ymuli = 4.56968499978;
  var bmuli = 11.3187123616;
  var y_to_b_muli = -0.634288116438;

  var xmul=(xmuli);
  var ymul=(ymuli);
  var bmul=(bmuli);
  var y_to_b_mul=(y_to_b_muli);
  var b = b_arg + y_to_b_mul * y;
  valb[0] = b * bmul;
  valx[0] = x * xmul;
  valy[0] = y * ymul;
}

//491
function SuppressHfInBrightAreas(xsize, ysize,
                                 hf,
                                 brightness) {
  var inew=new ImageF();inew._2(xsize, ysize);
  var mul = 1.10176291616;
  var mul2 = 3.0563595934;
  var reg = 2000 * mul2;
  for (var y = 0; y < ysize; ++y) {
    var rowhf = hf.Row(y);var rowhf_off = hf.Row_off(y);
    var rowbr = brightness.Row(y);var rowbr_off = brightness.Row_off(y);
    var rownew = inew.Row(y);var rownew_off = inew.Row_off(y);
    for (var x = 0; x < xsize; ++x) {
      var v = rowhf[rowhf_off+x];
      var scaler = mul * reg / (reg + rowbr[rowbr_off+x]);
      rownew[rownew_off+x] = scaler * v;
    }
  }
  return inew;
}


//512
function MaximumClamping(xsize, ysize, ix,
                         yw) {
  var inew=new ImageF();inew._2(xsize, ysize);
  for (var y = 0; y < ysize; ++y) {
    var rowx = ix.Row(y);var rowx_off = ix.Row_off(y);
    var rownew = inew.Row(y);var rownew_off = inew.Row_off(y);
    for (var x = 0; x < xsize; ++x) {
      var v = rowx[rowx_off+x];
      if (v >= yw) {
        v -= yw;
        v *= 0.7;
        v += yw;
      } else if (v < -yw) {
        v += yw;
        v *= 0.7;
        v -= yw;
      }
      rownew[rownew_off+x] = v;
    }
  }
  return inew;
}

//535
function Suppress(x, y) {
  var yw = 14.7257226847;
  var s = 0.536690340523;
  var scaler = s + (yw * (1.0 - s)) / (yw + y * y);
  return scaler * x;
}

//542
function SuppressXByY(xsize, ysize,
                      ix, iy, w) {
  var inew=new ImageF();inew._2(xsize, ysize);
  for (var y = 0; y < ysize; ++y) {
    var rowx = ix.Row(y);var rowx_off = ix.Row_off(y);
    var rowy = iy.Row(y);var rowy_off = iy.Row_off(y);
    var rownew = inew.Row(y);var rownew_off = inew.Row_off(y);
    for (var x = 0; x < xsize; ++x) {
      rownew[rownew_off+x] = Suppress(rowx[rowx_off+x], w * rowy[rowy_off+x]);
    }
  }
  return inew;
}

//556
function SeparateFrequencies(
    xsize, ysize,
    xyb,
    ps) {
  //PROFILER_FUNC;
  ps.lf.length=(3);
  ps.mf.length=(3);
  ps.hf.length=(3);
  ps.uhf.length=(3);
  for (var i = 0; i < 3; ++i) {
    // Extract lf ...
    var kSigmaLf = 7.549782202;
    ps.lf[i] = DoGBlur(xyb[i], kSigmaLf, 0.0);
    // ... and keep everything else in mf.
    ps.mf[i] = new ImageF();ps.mf[i]._2(xsize, ysize);
    for (var y = 0; y < ysize; ++y) {
      for (var x = 0; x < xsize; ++x) {
        ps.mf[i].Row(y)[ps.mf[i].Row_off(y)+x] = xyb[i].Row(y)[xyb[i].Row_off(y)+x] - ps.lf[i].Row(y)[ps.lf[i].Row_off(y)+x];
      }
    }
    // Divide mf into mf and hf.
    var kSigmaHf = 0.5 * kSigmaLf;
    ps.hf[i] = new ImageF();ps.hf[i]._2(xsize, ysize);
    for (var y = 0; y < ysize; ++y) {
      for (var x = 0; x < xsize; ++x) {
        ps.hf[i].Row(y)[ps.hf[i].Row_off(y)+x] = ps.mf[i].Row(y)[ps.mf[i].Row_off(y)+x];
      }
    }
    ps.mf[i] = DoGBlur(ps.mf[i], kSigmaHf, 0.0);
    for (var y = 0; y < ysize; ++y) {
      for (var x = 0; x < xsize; ++x) {
        ps.hf[i].Row(y)[ps.hf[i].Row_off(y)+x] -= ps.mf[i].Row(y)[ps.mf[i].Row_off(y)+x];
      }
    }
    // Divide hf into hf and uhf.
    var kSigmaUhf = 0.5 * kSigmaHf;
    ps.uhf[i] = new ImageF();ps.uhf[i]._2(xsize, ysize);
    for (var y = 0; y < ysize; ++y) {
      for (var x = 0; x < xsize; ++x) {
        ps.uhf[i].Row(y)[ps.uhf[i].Row_off(y)+x] = ps.hf[i].Row(y)[ps.hf[i].Row_off(y)+x];
      }
    }
    ps.hf[i] = DoGBlur(ps.hf[i], kSigmaUhf, 0.0);
    for (var y = 0; y < ysize; ++y) {
      for (var x = 0; x < xsize; ++x) {
        ps.uhf[i].Row(y)[ps.uhf[i].Row_off(y)+x] -= ps.hf[i].Row(y)[ps.hf[i].Row_off(y)+x];
      }
    }
  }
  // Modify range around zero code only concerns the high frequency
  // planes and only the X and Y channels.
  var uhf_xy_modification = [//[2]
    -0.112117785772,
    -4.858045964
  ];
  var hf_xy_modification = [//[2]
    0.0323432253707,
    -0.0533891565408
  ];
  var mf_xy_modification = [//[2]
    0.0181347124804,
    -0.126105706599
  ];
  ps.uhf = ModifyRangeAroundZero(uhf_xy_modification, ps.uhf);
  ps.hf = ModifyRangeAroundZero(hf_xy_modification, ps.hf);
  ps.mf = ModifyRangeAroundZero(mf_xy_modification, ps.mf);
  // Convert low freq xyb to vals space so that we can do a simple squared sum
  // diff on the low frequencies later.
  for (var y = 0; y < ysize; ++y) {
    var row_x = ps.lf[0].Row(y);var row_x_off = ps.lf[0].Row_off(y);
    var row_y = ps.lf[1].Row(y);var row_y_off = ps.lf[1].Row_off(y);
    var row_b = ps.lf[2].Row(y);var row_b_off = ps.lf[2].Row_off(y);
    for (var x = 0; x < xsize; ++x) {
      var valx=[], valy=[], valb=[];
      XybLowFreqToVals(row_x[row_x_off+x], row_y[row_y_off+x], row_b[row_b_off+x], valx, valy, valb);//& & &
      row_x[row_x_off+x] = valx[0];
      row_y[row_y_off+x] = valy[0];
      row_b[row_b_off+x] = valb[0];
    }
  }
if(1) {
  // Suppress red-green by intensity change.
  var suppress = [//[3]
    0.0123070791057,
    28.0842005311
  ];
  ps.uhf[0] = SuppressXByY(xsize, ysize, ps.uhf[0], ps.uhf[1], suppress[0]);
  ps.hf[0] = SuppressXByY(xsize, ysize, ps.hf[0], ps.hf[1], suppress[1]);
  var maxclamp0 = 0.764101528619;
  ps.uhf[0] = MaximumClamping(xsize, ysize, ps.uhf[0], maxclamp0);
  var maxclamp1 = 2.63290517726;
  ps.hf[0] = MaximumClamping(xsize, ysize, ps.hf[0], maxclamp1);
  var maxclamp2 = 54.4858042922;
  ps.uhf[1] = MaximumClamping(xsize, ysize, ps.uhf[1], maxclamp2);
  var maxclamp3 = 41.3578204305;
  ps.hf[1] = MaximumClamping(xsize, ysize, ps.hf[1], maxclamp3);

  ps.hf[1] = SuppressHfInBrightAreas(xsize, ysize, ps.hf[1], ps.lf[1]);
  ps.uhf[1] = SuppressHfInBrightAreas(xsize, ysize, ps.uhf[1], ps.lf[1]);
  ps.mf[1] = SuppressHfInBrightAreas(xsize, ysize, ps.mf[1], ps.lf[1]);
}
/*if(0) {
  DumpPpm("/tmp/psorig.ppm", xyb);
  DumpPpm("/tmp/ps0.ppm", ps.lf);
  DumpPpm("/tmp/ps1.ppm", ps.mf);
  DumpPpm("/tmp/ps2.ppm", ps.hf);
  DumpPpm("/tmp/ps3.ppm", ps.uhf);
}*/
}

//666
function SameNoiseLevels(i0, i1,
                         kSigma,
                         w,
                         maxclamp,
                         diffmap) {
  var blurred0 = CopyPixels(i0);
  var blurred1 = CopyPixels(i1);
  for (var y = 0; y < i0.ysize(); ++y) {
    var row0 = blurred0.Row(y);var row0_off = blurred0.Row_off(y);
    var row1 = blurred1.Row(y);var row1_off = blurred1.Row_off(y);
    for (var x = 0; x < i0.xsize(); ++x) {
      row0[row0_off+x] = fabs(row0[row0_off+x]);
      row1[row1_off+x] = fabs(row1[row1_off+x]);
      if (row0[row0_off+x] > maxclamp) row0[row0_off+x] = maxclamp;
      if (row1[row1_off+x] > maxclamp) row1[row1_off+x] = maxclamp;
    }
  }
  blurred0 = Blur(blurred0, kSigma, 0.0);
  blurred1 = Blur(blurred1, kSigma, 0.0);
  for (var y = 0; y < i0.ysize(); ++y) {
    var row0 = blurred0.Row(y);var row0_off = blurred0.Row_off(y);
    var row1 = blurred1.Row(y);var row1_off = blurred1.Row_off(y);
    var row_diff = diffmap.Row(y);var row_diff_off = diffmap.Row_off(y);
    for (var x = 0; x < i0.xsize(); ++x) {
      var diff = row0[row0_off+x] - row1[row1_off+x];
      row_diff[row_diff_off+x] += w * diff * diff;
    }
  }
}

//696
function SameNoiseLevelsX(i0, i1,
                           kSigma,
                           w,
                           maxclamp,
                           diffmap) {
  var blurred0 = CopyPixels(i0);
  var blurred1 = CopyPixels(i1);
  for (var y = 0; y < i0.ysize(); ++y) {
    var row0 = blurred0.Row(y);var row0_off = blurred0.Row_off(y);
    var row1 = blurred1.Row(y);var row1_off = blurred1.Row_off(y);
    for (var x = i0.xsize() - 1; x != 0; --x) {
      row0[row0_off+x] -= row0[row0_off+x - 1];
      row1[row1_off+x] -= row1[row1_off+x - 1];
      row0[row0_off+x] = fabs(row0[row0_off+x]);
      row1[row1_off+x] = fabs(row1[row1_off+x]);
      if (row0[row0_off+x] > maxclamp) row0[row0_off+x] = maxclamp;
      if (row1[row1_off+x] > maxclamp) row1[row1_off+x] = maxclamp;
    }
    row0[row0_off+0] = 0.25 * row0[row0_off+1];
    row1[row1_off+0] = 0.25 * row0[row0_off+1];
  }
  blurred0 = Blur(blurred0, kSigma, 0.0);
  blurred1 = Blur(blurred1, kSigma, 0.0);
  for (var y = 0; y < i0.ysize(); ++y) {
    var row0 = blurred0.Row(y);var row0_off = blurred0.Row_off(y);
    var row1 = blurred1.Row(y);var row1_off = blurred1.Row_off(y);
    var row_diff = diffmap.Row(y);var row_diff_off = diffmap.Row_off(y);
    for (var x = 0; x < i0.xsize(); ++x) {
      var diff = row0[row0_off+x] - row1[row1_off+x];
      row_diff[row_diff_off+x] += w * diff * diff;
    }
  }
}
//730
function SameNoiseLevelsY(i0, i1,
                          kSigma,
                          w,
                          maxclamp,
                          diffmap) {
  var blurred0 = CopyPixels(i0);
  var blurred1 = CopyPixels(i1);
  for (var y = i0.ysize() - 1; y != 0; --y) {
    var row0prev = blurred0.Row(y - 1);var row0prev_off = blurred0.Row_off(y - 1);
    var row1prev = blurred1.Row(y - 1);var row1prev_off = blurred1.Row_off(y - 1);
    var row0 = blurred0.Row(y);var row0_off = blurred0.Row_off(y);
    var row1 = blurred1.Row(y);var row1_off = blurred1.Row_off(y);
    for (var x = 0; x < i0.xsize(); ++x) {
      row0[row0_off+x] -= row0prev[row0prev_off+x];
      row1[row1_off+x] -= row1prev[row1prev_off+x];
      row0[row0_off+x] = fabs(row0[row0_off+x]);
      row1[row1_off+x] = fabs(row1[row1_off+x]);
      if (row0[row0_off+x] > maxclamp) row0[row0_off+x] = maxclamp;
      if (row1[row1_off+x] > maxclamp) row1[row1_off+x] = maxclamp;
    }
  }
  {
    var row0 = blurred0.Row(0);var row0_off = blurred0.Row_off(0);
    var row1 = blurred1.Row(0);var row1_off = blurred1.Row_off(0);
    var row0next = blurred0.Row(1);var row0next_off = blurred0.Row_off(1);
    var row1next = blurred1.Row(1);var row1next_off = blurred1.Row_off(1);
    for (var x = 0; x < i0.xsize(); ++x) {
      row0[row0_off+x] = 0.25 * row0next[row0next_off+x];
      row1[row1_off+x] = 0.25 * row1next[row1next_off+x];
    }
  }
  blurred0 = Blur(blurred0, kSigma, 0.0);
  blurred1 = Blur(blurred1, kSigma, 0.0);
  for (var y = 0; y < i0.ysize(); ++y) {
    var row0 = blurred0.Row(y);var row0_off = blurred0.Row_off(y);
    var row1 = blurred1.Row(y);var row1_off = blurred1.Row_off(y);
    var row_diff = diffmap.Row(y);var row_diff_off = diffmap.Row_off(y);
    for (var x = 0; x < i0.xsize(); ++x) {
      var diff = row0[row0_off+x] - row1[row1_off+x];
      row_diff[row_diff_off+x] += w * diff * diff;
    }
  }
}

//774
function SameNoiseLevelsYP1(i0, i1,
                            kSigma,
                            w,
                            maxclamp,
                            diffmap) {
  var blurred0 = CopyPixels(i0);
  var blurred1 = CopyPixels(i1);
  for (var y = i0.ysize() - 1; y != 0; --y) {
    var row0prev = blurred0.Row(y - 1);var row0prev_off = blurred0.Row_off(y - 1);
    var row1prev = blurred1.Row(y - 1);var row1prev_off = blurred1.Row_off(y - 1);
    var row0 = blurred0.Row(y);var row0_off = blurred0.Row_off(y);
    var row1 = blurred1.Row(y);var row1_off = blurred1.Row_off(y);
    for (var x = 1; x < i0.xsize(); ++x) {
      row0[row0_off+x] -= row0prev[row0prev_off+x - 1];
      row1[row1_off+x] -= row1prev[row1prev_off+x - 1];
      row0[row0_off+x] = fabs(row0[row0_off+x]);
      row1[row1_off+x] = fabs(row1[row1_off+x]);
      if (row0[row0_off+x] > maxclamp) row0[row0_off+x] = maxclamp;
      if (row1[row1_off+x] > maxclamp) row1[row1_off+x] = maxclamp;
    }
    row0[row0_off+0] = 0.25 * row0[row0_off+1];
    row1[row1_off+0] = 0.25 * row1[row1_off+1];
  }
  {
    var row0 = blurred0.Row(0);var row0_off = blurred0.Row_off(0);
    var row1 = blurred1.Row(0);var row1_off = blurred1.Row_off(0);
    var row0next = blurred0.Row(1);var row0next_off = blurred0.Row_off(1);
    var row1next = blurred1.Row(1);var row1next_off = blurred1.Row_off(1);
    for (var x = 0; x < i0.xsize(); ++x) {
      row0[row0_off+x] = 0.25 * row0next[row0next_off+x];
      row1[row1_off+x] = 0.25 * row1next[row1next_off+x];
    }
  }
  blurred0 = Blur(blurred0, kSigma, 0.0);
  blurred1 = Blur(blurred1, kSigma, 0.0);
  for (var y = 0; y < i0.ysize(); ++y) {
    var row0 = blurred0.Row(y);var row0_off = blurred0.Row_off(y);
    var row1 = blurred1.Row(y);var row1_off = blurred1.Row_off(y);
    var row_diff = diffmap.Row(y);var row_diff_off = diffmap.Row_off(y);
    for (var x = 0; x < i0.xsize(); ++x) {
      var diff = row0[row0_off+x] - row1[row1_off+x];
      row_diff[row_diff_off+x] += w * diff * diff;
    }
  }
}


//820
function SameNoiseLevelsYM1(i0, i1,
                            kSigma,
                            w,
                            maxclamp,
                            diffmap) {
  var blurred0 = CopyPixels(i0);
  var blurred1 = CopyPixels(i1);
  for (var y = i0.ysize() - 1; y != 0; --y) {
    var row0prev = blurred0.Row(y - 1);var row0prev_off = blurred0.Row_off(y - 1);
    var row1prev = blurred1.Row(y - 1);var row1prev_off = blurred1.Row_off(y - 1);
    var row0 = blurred0.Row(y);var row0_off = blurred0.Row_off(y);
    var row1 = blurred1.Row(y);var row1_off = blurred1.Row_off(y);
    for (var x = 0; x + 1 < i0.xsize(); ++x) {
      row0[row0_off+x] -= row0prev[row0prev_off+x + 1];
      row1[row1_off+x] -= row1prev[row1prev_off+x + 1];
      row0[row0_off+x] = fabs(row0[row0_off+x]);
      row1[row1_off+x] = fabs(row1[row1_off+x]);
      if (row0[row0_off+x] > maxclamp) row0[row0_off+x] = maxclamp;
      if (row1[row1_off+x] > maxclamp) row1[row1_off+x] = maxclamp;
    }
    row0[row0_off+i0.xsize() - 1] = 0.25 * row0[row0_off+i0.xsize() - 2];
    row1[row1_off+i0.xsize() - 1] = 0.25 * row1[row1_off+i0.xsize() - 2];
  }
  {
    var row0 = blurred0.Row(0);var row0_off = blurred0.Row_off(0);
    var row1 = blurred1.Row(0);var row1_off = blurred1.Row_off(0);
    var row0next = blurred0.Row(1);var row0next_off = blurred0.Row_off(1);
    var row1next = blurred1.Row(1);var row1next_off = blurred1.Row_off(1);
    for (var x = 0; x < i0.xsize(); ++x) {
      row0[row0_off+x] = 0.25 * row0next[row0next_off+x];
      row1[row1_off+x] = 0.25 * row1next[row1next_off+x];
    }
  }
  blurred0 = Blur(blurred0, kSigma, 0.0);
  blurred1 = Blur(blurred1, kSigma, 0.0);
  for (var y = 0; y < i0.ysize(); ++y) {
    var row0 = blurred0.Row(y);var row0_off = blurred0.Row_off(y);
    var row1 = blurred1.Row(y);var row1_off = blurred1.Row_off(y);
    var row_diff = diffmap.Row(y);var row_diff_off = diffmap.Row_off(y);
    for (var x = 0; x < i0.xsize(); ++x) {
      var diff = row0[row0_off+x] - row1[row1_off+x];
      row_diff[row_diff_off+x] += w * diff * diff;
    }
  }
}
//880
function LNDiff(i0, i1, w,
                n,
                diffmap) {
  if (n == 1.0) {
    for (var y = 0; y < i0.ysize(); ++y) {
      var row0 = i0.Row(y);var row0_off = i0.Row_off(y);
      var row1 = i1.Row(y);var row1_off = i1.Row_off(y);
      var row_diff = diffmap.Row(y);var row_diff_off = diffmap.Row_off(y);
      for (var x = 0; x < i0.xsize(); ++x) {
        var diff = fabs(row0[row0_off+x] - row1[row1_off+x]);
        row_diff[row_diff_off+x] += w * diff;
      }
    }
  } else if (n == 2.0) {
    for (var y = 0; y < i0.ysize(); ++y) {
      var row0 = i0.Row(y);var row0_off = i0.Row_off(y);
      var row1 = i1.Row(y);var row1_off = i1.Row_off(y);
      var row_diff = diffmap.Row(y);var row_diff_off = diffmap.Row_off(y);
      for (var x = 0; x < i0.xsize(); ++x) {
        var diff = row0[row0_off+x] - row1[row1_off+x];
        row_diff[row_diff_off+x] += w * diff * diff;
      }
    }
  } else {
    for (var y = 0; y < i0.ysize(); ++y) {
      var row0 = i0.Row(y);var row0_off = i0.Row_off(y);
      var row1 = i1.Row(y);var row1_off = i1.Row_off(y);
      var row_diff = diffmap.Row(y);var row_diff_off = diffmap.Row_off(y);
      for (var x = 0; x < i0.xsize(); ++x) {
        var diff = fabs(row0[row0_off+x] - row1[row1_off+x]);
        row_diff[row_diff_off+x] += w * Math.pow(diff, n);
      }
    }
  }
}

//916
// Making a cluster of local errors to be more impactful than
// just a single error.
function CalculateDiffmap(diffmap_in) {
  //PROFILER_FUNC;
  // Take square root.
  var diffmap=new ImageF();diffmap._2(diffmap_in.xsize(), diffmap_in.ysize());
  var kInitialSlope = 100.0;
  for (var y = 0; y < diffmap.ysize(); ++y) {
    var row_in = diffmap_in.Row(y);var row_in_off = diffmap_in.Row_off(y);
    var row_out = diffmap.Row(y);var row_out_off = diffmap.Row_off(y);
    for (var x = 0; x < diffmap.xsize(); ++x) {
      var orig_val = row_in[row_in_off+x];
      // TODO(b/29974893): Until that is fixed do not call sqrt on very small
      // numbers.
      row_out[row_out_off+x] = (orig_val < (1.0 / (kInitialSlope * kInitialSlope))
                    ? kInitialSlope * orig_val
                    : Math.sqrt(orig_val));
    }
  }
  {
    var kSigma = 1.63788154557;
    var mul1 = 0.537065242152;
    var scale = 1.0 / (1.0 + mul1);
    var border_ratio = 1.0; // 2.01209066992;
    var blurred = Blur(diffmap, kSigma, border_ratio);
    for (var y = 0; y < diffmap.ysize(); ++y) {
      var row_blurred = blurred.Row(y);var row_blurred_off = blurred.Row_off(y);
      var row = diffmap.Row(y);var row_off = diffmap.Row_off(y);
      for (var x = 0; x < diffmap.xsize(); ++x) {
        row[row_off+x] += mul1 * row_blurred[row_blurred_off+x];
        row[row_off+x] *= scale;
      }
    }
  }
  return diffmap;
}

//953
function MaskPsychoImage(pi0, pi1,
                     xsize, ysize,
                     mask,
                     mask_dc) {
  var mask_xyb0 = CreatePlanes(xsize, ysize, 3);//std::vector<ImageF>
  var mask_xyb1 = CreatePlanes(xsize, ysize, 3);
  var muls = [//[4]
    -2.60009303e-06,
    1.37122152964,
    0.925101999009,
    2.47562554208
  ];
  for (var i = 0; i < 2; ++i) {
    var a = muls[2 * i];
    var b = muls[2 * i + 1];
    for (var y = 0; y < ysize; ++y) {
      var row_hf0 = pi0.hf[i].Row(y);var row_hf0_off = pi0.hf[i].Row_off(y);
      var row_hf1 = pi1.hf[i].Row(y);var row_hf1_off = pi1.hf[i].Row_off(y);
      var row_uhf0 = pi0.uhf[i].Row(y);var row_uhf0_off = pi0.uhf[i].Row_off(y);
      var row_uhf1 = pi1.uhf[i].Row(y);var row_uhf1_off = pi1.uhf[i].Row_off(y);
      var row0 = mask_xyb0[i].Row(y);var row0_off = mask_xyb0[i].Row_off(y);
      var row1 = mask_xyb1[i].Row(y);var row1_off = mask_xyb1[i].Row_off(y);
      for (var x = 0; x < xsize; ++x) {
        row0[row0_off+x] = a * row_uhf0[row_uhf0_off+x] + b * row_hf0[row_hf0_off+x];
        row1[row1_off+x] = a * row_uhf1[row_uhf1_off+x] + b * row_hf1[row_hf1_off+x];
      }
    }
  }
  Mask(mask_xyb0, mask_xyb1, mask, mask_dc);
}


//984
function ButteraugliComparator(rgb0) {
  this.xsize_=0;  
  this.ysize_=0;  
  this.num_pixels_=0;  
  this.pi0_=new PsychoImage();
      this.xsize_=(rgb0[0].xsize()),      
	  this.ysize_=(rgb0[0].ysize()),      
	  this.xnum_pixels_=(this.xsize_ * this.ysize_);  
	  if (this.xsize_ < 8 || this.ysize_ < 8) return;  
	  var xyb0 = OpsinDynamicsImage(rgb0);  
	  SeparateFrequencies(this.xsize_, this.ysize_, xyb0, this.pi0_);
}
//1002
ButteraugliComparator.prototype.Diffmap=function(rgb1,
                                    result) {
  //PROFILER_FUNC;  
  if (this.xsize_ < 8 || this.ysize_ < 8) return;  
  this.DiffmapOpsinDynamicsImage(OpsinDynamicsImage(rgb1), result);
}
//1009
ButteraugliComparator.prototype.DiffmapOpsinDynamicsImage=function(
    xyb1,
    result) {
  //PROFILER_FUNC;
  if (this.xsize_ < 8 || this.ysize_ < 8) return;
  var pi1=new PsychoImage();
  SeparateFrequencies(this.xsize_, this.ysize_, xyb1, pi1);
/*if(0) {
  DumpPpm("/tmp/sep1_orig.ppm", xyb1);
}*/
  result[0] = new ImageF();
  result[0]._2(this.xsize_, this.ysize_);
  this.DiffmapPsychoImage(pi1, result);
}
//1023
ButteraugliComparator.prototype.DiffmapPsychoImage=function(pi1,
                                               result) {
  //PROFILER_FUNC;
  if (this.xsize_ < 8 || this.ysize_ < 8) {
    return;
  }
/*if(0) {
  PrintStatistics("hf0", this.pi0_.hf);
  PrintStatistics("hf1", this.pi1.hf);
  PrintStatistics("mf0", this.pi0_.mf);
  PrintStatistics("mf1", this.pi1.mf);
  PrintStatistics("lf0", this.pi0_.lf);
  PrintStatistics("lf1", this.pi1.lf);
}*/

  var block_diff_dc=mallocArrOI(3,ImageF);
  var block_diff_ac=mallocArrOI(3,ImageF);
  for (var c = 0; c < 3; ++c) {
    block_diff_dc[c] = new ImageF();block_diff_dc[c]._3(this.xsize_, this.ysize_, 0.0);
    block_diff_ac[c] = new ImageF();block_diff_ac[c]._3(this.xsize_, this.ysize_, 0.0);
  }

  var wUhfMalta = 1.37246434724;
  var norm1Uhf = 500;
  this.MaltaDiffMap(this.pi0_.uhf[1], pi1.uhf[1], wUhfMalta, norm1Uhf,
               block_diff_ac[1]);//&

  var wUhfMaltaX = 2.22301383652;
  var norm1UhfX = 500;
  this.MaltaDiffMap(this.pi0_.uhf[0], pi1.uhf[0], wUhfMaltaX, norm1UhfX,
               block_diff_ac[0]);

  var wHfMalta = 10.7925139357;
  var norm1Hf = 500;
  this.MaltaDiffMap(this.pi0_.hf[1], pi1.hf[1], wHfMalta, norm1Hf,
               block_diff_ac[1]);

  var wHfMaltaX = 164.899610068;
  var norm1HfX = 500;
  this.MaltaDiffMap(this.pi0_.hf[0], pi1.hf[0], wHfMaltaX, norm1HfX,
               block_diff_ac[0]);

  var wMfMaltaX = 46.2757637244;
  var norm1MfX = 500;
  this.MaltaDiffMap(this.pi0_.mf[0], pi1.mf[0], wMfMaltaX, norm1MfX,
               block_diff_ac[0]);

  var wmul = [//[13]
    0,
    6.41600396696,
    0,
    0,
    7.04885534151,
    14.8131395219,
    0.908141790141,
    5.58430520339,
    1.77583623158,
    0.0,
    79.5154856641,
    0,
    74.1571576582
  ];


  var kSigmaHf = 9.67477693518;
  var maxclamp = 30.0946403403;
  SameNoiseLevels(this.pi0_.hf[1], pi1.hf[1], kSigmaHf, wmul[10], maxclamp,
                  block_diff_ac[1]);//&
  SameNoiseLevels(this.pi0_.hf[1], pi1.hf[1], 0.5 * kSigmaHf, wmul[11], maxclamp,
                  block_diff_ac[1]);
  var kSigmaHfX = 8.47017380014;
  SameNoiseLevelsX(this.pi0_.hf[1], pi1.hf[1], kSigmaHfX, wmul[12], maxclamp,
                   block_diff_ac[1]);
  SameNoiseLevelsY(this.pi0_.hf[1], pi1.hf[1], kSigmaHfX, wmul[12], maxclamp,
                   block_diff_ac[1]);
  SameNoiseLevelsYP1(this.pi0_.hf[1], pi1.hf[1], kSigmaHfX, wmul[12], maxclamp,
                     block_diff_ac[1]);
  SameNoiseLevelsYM1(this.pi0_.hf[1], pi1.hf[1], kSigmaHfX, wmul[12], maxclamp,
                     block_diff_ac[1]);


  var valn = [//[9]
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    2.0,
    2.0,
    2.0,
    2.0
  ];

  for (var c = 0; c < 3; ++c) {
    if (wmul[c] != 0) {
      LNDiff(this.pi0_.hf[c], pi1.hf[c], wmul[c], valn[c], block_diff_ac[c]);//&
    }
    LNDiff(this.pi0_.mf[c], pi1.mf[c], wmul[3 + c], valn[3 + c], block_diff_ac[c]);
    LNDiff(this.pi0_.lf[c], pi1.lf[c], wmul[6 + c], valn[6 + c], block_diff_dc[c]);
  }

/*if(0) {
  // Doesn't really work for now.
  var wBlueCorr = 0.0;
  var blurred_b_y_correlation0 = BlurredBlueCorrelation(pi0_.uhf, this.pi0_.hf);
  var blurred_b_y_correlation1 = BlurredBlueCorrelation(pi1.uhf, this.pi1.hf);
  L2Diff(blurred_b_y_correlation0, blurred_b_y_correlation1, wBlueCorr,
         block_diff_ac[2]);
}*/

  var mask_xyb=[];//std::vector<ImageF>
  var mask_xyb_dc=[];
  MaskPsychoImage(this.pi0_, pi1, this.xsize_, this.ysize_, mask_xyb, mask_xyb_dc);//& &

/*if(0) {
  DumpPpm("/tmp/mask.ppm", mask_xyb, 777);
}*/

  result[0] = CalculateDiffmap(
      this.CombineChannels(mask_xyb, mask_xyb_dc[0], block_diff_dc, block_diff_ac));
/*if(0) {
  PrintStatistics("diffmap", result);
}*/
}
function MaltaUnit(d, d_off, xs) {
  var xs3 = 3 * xs;
  var retval = 0;
  var kEdgemul = 0.0736824429946;
  {
    // x grows, y constant
    var sum =
        d[d_off-4] +
        d[d_off-3] +
        d[d_off-2] +
        d[d_off-1] +
        d[d_off+0] +
        d[d_off+1] +
        d[d_off+2] +
        d[d_off+3] +
        d[d_off+4];
    retval += sum * sum;
    var sum2 =
        d[d_off+xs - 4] +
        d[d_off+xs - 3] +
        d[d_off+xs - 2] +
        d[d_off+xs - 1] +
        d[d_off+xs] +
        d[d_off+xs + 1] +
        d[d_off+xs + 2] +
        d[d_off+xs + 3] +
        d[d_off+xs + 4];
    var edge = sum - sum2;
    retval += kEdgemul * edge * edge;
  }
  {
    // y grows, x constant
    var sum =
        d[d_off-xs3 - xs] +
        d[d_off-xs3] +
        d[d_off-xs - xs] +
        d[d_off-xs] +
        d[d_off+0] +
        d[d_off+xs] +
        d[d_off+xs + xs] +
        d[d_off+xs3] +
        d[d_off+xs3 + xs];
    retval += sum * sum;
    var sum2 =
        d[d_off-xs3 - xs + 1] +
        d[d_off-xs3 + 1] +
        d[d_off-xs - xs + 1] +
        d[d_off-xs + 1] +
        d[d_off+1] +
        d[d_off+xs + 1] +
        d[d_off+xs + xs + 1] +
        d[d_off+xs3 + 1] +
        d[d_off+xs3 + xs + 1];
    var edge = sum - sum2;
    retval += kEdgemul * edge * edge;
  }
  {
    // both grow
    var sum =
        d[d_off-xs3 - 3] +
        d[d_off-xs - xs - 2] +
        d[d_off-xs - 1] +
        d[d_off+0] +
        d[d_off+xs + 1] +
        d[d_off+xs + xs + 2] +
        d[d_off+xs3 + 3];
    retval += sum * sum;
  }
  {
    // y grows, x shrinks
    var sum =
        d[d_off-xs3 + 3] +
        d[d_off-xs - xs + 2] +
        d[d_off-xs + 1] +
        d[d_off+0] +
        d[d_off+xs - 1] +
        d[d_off+xs + xs - 2] +
        d[d_off+xs3 - 3];
    retval += sum * sum;
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    var sum =
        d[d_off-xs3 - xs + 1] +
        d[d_off-xs3 + 1] +
        d[d_off-xs - xs + 1] +
        d[d_off-xs] +
        d[d_off+0] +
        d[d_off+xs] +
        d[d_off+xs - 1] +
        d[d_off+xs3 - 1] +
        d[d_off+xs3 + xs - 1];
    retval += sum * sum;
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    var sum =
        d[d_off-xs3 - xs - 1] +
        d[d_off-xs3 - 1] +
        d[d_off-xs - xs - 1] +
        d[d_off-xs] +
        d[d_off+0] +
        d[d_off+xs] +
        d[d_off+xs + 1] +
        d[d_off+xs3 + 1] +
        d[d_off+xs3 + xs + 1];
    retval += sum * sum;
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    var sum =
        d[d_off-4 - xs] +
        d[d_off-3 - xs] +
        d[d_off-2 - xs] +
        d[d_off-1] +
        d[d_off+0] +
        d[d_off+1] +
        d[d_off+2 + xs] +
        d[d_off+3 + xs] +
        d[d_off+4 + xs];
    retval += sum * sum;
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    var sum =
        d[d_off-4 + xs] +
        d[d_off-3 + xs] +
        d[d_off-2 + xs] +
        d[d_off-1] +
        d[d_off+0] +
        d[d_off+1] +
        d[d_off+2 - xs] +
        d[d_off+3 - xs] +
        d[d_off+4 - xs];
    retval += sum * sum;
  }
  {
    /* 0_________
       1__*______
       2___*_____
       3___*_____
       4____0____
       5_____*___
       6_____*___
       7______*__
       8_________ */
    var sum =
        d[d_off-xs3 - 2] +
        d[d_off-xs - xs - 1] +
        d[d_off-xs - 1] +
        d[d_off+0] +
        d[d_off+xs + 1] +
        d[d_off+xs + xs + 1] +
        d[d_off+xs3 + 2];
    retval += sum * sum;
  }
  {
    /* 0_________
       1______*__
       2_____*___
       3_____*___
       4____0____
       5___*_____
       6___*_____
       7__*______
       8_________ */
    var sum =
        d[d_off-xs3 + 2] +
        d[d_off-xs - xs + 1] +
        d[d_off-xs + 1] +
        d[d_off+0] +
        d[d_off+xs - 1] +
        d[d_off+xs + xs - 1] +
        d[d_off+xs3 - 2];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2_*_______
       3__**_____
       4____0____
       5_____**__
       6_______*_
       7_________
       8_________ */
    var sum =
        d[d_off-xs - xs - 3] +
        d[d_off-xs - 2] +
        d[d_off-xs - 1] +
        d[d_off+0] +
        d[d_off+xs + 1] +
        d[d_off+xs + 2] +
        d[d_off+xs + xs + 3];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2_______*_
       3_____**__
       4____0____
       5__**_____
       6_*_______
       7_________
       8_________ */
    var sum =
        d[d_off-xs - xs + 3] +
        d[d_off-xs + 2] +
        d[d_off-xs + 1] +
        d[d_off+0] +
        d[d_off+xs - 1] +
        d[d_off+xs - 2] +
        d[d_off+xs + xs - 3];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2_________
       3______**_
       4____0*___
       5__**_____
       6**_______
       7_________
       8_________ */

    var sum =
        d[d_off+xs + xs - 4] +
        d[d_off+xs + xs - 3] +
        d[d_off+xs - 2] +
        d[d_off+xs - 1] +
        d[d_off+0] +
        d[d_off+1] +
        d[d_off-xs + 2] +
        d[d_off-xs + 3];
    retval += sum * sum;
  }
  {
    /* 0_________
       1_________
       2**_______
       3__**_____
       4____0*___
       5______**_
       6_________
       7_________
       8_________ */
    var sum =
        d[d_off-xs - xs - 4] +
        d[d_off-xs - xs - 3] +
        d[d_off-xs - 2] +
        d[d_off-xs - 1] +
        d[d_off+0] +
        d[d_off+1] +
        d[d_off+xs + 2] +
        d[d_off+xs + 3];
    retval += sum * sum;
  }
  {
    /* 0__*______
       1__*______
       2___*_____
       3___*_____
       4____0____
       5____*____
       6_____*___
       7_____*___
       8_________ */
    var sum =
        d[d_off-xs3 - xs - 2] +
        d[d_off-xs3 - 2] +
        d[d_off-xs - xs - 1] +
        d[d_off-xs - 1] +
        d[d_off+0] +
        d[d_off+xs] +
        d[d_off+xs + xs + 1] +
        d[d_off+xs3 + 1];
    retval += sum * sum;
  }
  {
    /* 0______*__
       1______*__
       2_____*___
       3_____*___
       4____0____
       5____*____
       6___*_____
       7___*_____
       8_________ */
    var sum =
        d[d_off-xs3 - xs + 2] +
        d[d_off-xs3 + 2] +
        d[d_off-xs - xs + 1] +
        d[d_off-xs + 1] +
        d[d_off+0] +
        d[d_off+xs] +
        d[d_off+xs + xs - 1] +
        d[d_off+xs3 - 1];
    retval += sum * sum;
  }
  return retval;
}

//1452
ButteraugliComparator.prototype.MaltaDiffMap=function(
    y0, y1,
    weight,
    norm1,
    block_diff_ac) {
  //PROFILER_FUNC;
  var len = 3.75;
  var mulli = 0.414348163394;
  var w = mulli * Math.sqrt(weight) / (len * 2 + 1);
  var norm2 = w * norm1;
  var diffs=mallocArr(this.ysize_ * this.xsize_,0.0);
  var sums=mallocArr(this.ysize_ * this.xsize_,0.0);
  for (var y = 0, ix = 0; y < this.ysize_; ++y) {
    var row0 = y0.Row(y);var row0_off = y0.Row_off(y);
    var row1 = y1.Row(y);var row1_off = y1.Row_off(y);
    for (var x = 0; x < this.xsize_; ++x, ++ix) {
      var absval = 0.5 * (Math.abs(row0[row0_off+x]) + Math.abs(row1[row1_off+x]));
      var diff = row0[row0_off+x] - row1[row1_off+x];
      var scaler = norm2 / (norm1 + absval);
      diffs[ix] = scaler * diff;
    }
  }
  var borderimage=mallocArr(9 * 9,0.0);
  for (var y0 = 0; y0 < this.ysize_; ++y0) {
    var row_diff = block_diff_ac.Row(y0);var row_diff_off = block_diff_ac.Row_off(y0);
    var fastModeY = y0 >= 4 && y0 < this.ysize_ - 4;
    for (var x0 = 0; x0 < this.xsize_; ++x0) {
      var ix0 = y0 * this.xsize_ + x0;
      var d = diffs;var d_off = +ix0;//* &
      var fastModeX = x0 >= 4 && x0 < this.xsize_ - 4;
      if (fastModeY && fastModeX) {
        row_diff[row_diff_off+x0] += MaltaUnit(d, d_off, this.xsize_);
      } else {
        for (var dy = 0; dy < 9; ++dy) {
          var y = y0 + dy - 4;
          if (y < 0 || y >= this.ysize_) {
            for (var dx = 0; dx < 9; ++dx) {
              borderimage[dy * 9 + dx] = 0;
            }
          } else {
            for (var dx = 0; dx < 9; ++dx) {
              var x = x0 + dx - 4;
              if (x < 0 || x >= this.xsize_) {
                borderimage[dy * 9 + dx] = 0;
              } else {
                borderimage[dy * 9 + dx] = diffs[y * this.xsize_ + x];
              }
            }
          }
        }
        row_diff[row_diff_off+x0] += MaltaUnit(borderimage,4 * 9 + 4, 9);
      }
    }
  }
}
//1508
ButteraugliComparator.prototype.CombineChannels=function(
    mask_xyb,
    mask_xyb_dc,
    block_diff_dc,
    block_diff_ac) {
  //PROFILER_FUNC;
  var result=new ImageF();result._2(this.xsize_, this.ysize_);
  for (var y = 0; y < this.ysize_; ++y) {
    var row_out = result.Row(y);var row_out_off = result.Row_off(y);
    for (var x = 0; x < this.xsize_; ++x) {
      var mask=mallocArr(3,0.0);
      var dc_mask=mallocArr(3,0.0);
      var diff_dc=mallocArr(3,0.0);
      var diff_ac=mallocArr(3,0.0);
      for (var i = 0; i < 3; ++i) {
        mask[i] = mask_xyb[i].Row(y)[mask_xyb[i].Row_off(y)+x];
        dc_mask[i] = mask_xyb_dc[i].Row(y)[mask_xyb_dc[i].Row_off(y)+x];
        diff_dc[i] = block_diff_dc[i].Row(y)[block_diff_dc[i].Row_off(y)+x];
        diff_ac[i] = block_diff_ac[i].Row(y)[block_diff_ac[i].Row_off(y)+x];
      }
      row_out[row_out_off+x] = (DotProduct(diff_dc, dc_mask) + DotProduct(diff_ac, mask));
    }
  }
  return result;
}

//1534
function ButteraugliScoreFromDiffmap(diffmap) {
  //PROFILER_FUNC;
  var retval = 0.0;
  for (var y = 0; y < diffmap.ysize(); ++y) {
    var row = diffmap.Row(y);var row_off = diffmap.Row_off(y);
    for (var x = 0; x < diffmap.xsize(); ++x) {
      retval = Math.max(retval, row[row_off+x]);
    }
  }
  return retval;
}

// ===== Functions used by Mask only =====
//1549
function MakeMask(
    extmul, extoff,
    mul, offset,
    scaler) {
  var lut=mallocArr(512,0.0);//std::array<double, 512>
  for (var i = 0; i < lut.length; ++i) {
    var c = mul / ((0.01 * scaler * i) + offset);
    lut[i] = kGlobalScale * (1.0 + extmul * (c + extoff));
    if (lut[i] < 1e-5) {
      lut[i] = 1e-5;
    }
    assert(lut[i] >= 0.0);
    lut[i] *= lut[i];
  }
  return lut;
}

//1566
function MaskX(delta) {
  //PROFILER_FUNC;
  var extmul = 2.34519597358;
  var extoff = 1.76706832899;
  var offset = 0.36000980903;
  var scaler = 14.5339183386;
  var mul = 6.11337930116;
  var lut = //static const std::array<double, 512>
                MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut, lut.length, delta);
}

//1578
function MaskY(delta) {
  //PROFILER_FUNC;
  var extmul = 0.973432315281;
  var extoff = -0.56621175456;
  var offset = 1.40865158018;
  var scaler = 1.01481280596;
  var mul = 7.24741735412;
  var lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut, lut.length, delta);
}

//1590
function MaskDcX(delta) {
  //PROFILER_FUNC;
  var extmul = 13.0432939015;
  var extoff = 0.585668980608;
  var offset = 0.864712378003;
  var scaler = 519.45682322;
  var mul = 4.72871406401;
  var lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut, lut.length, delta);
}

//1602
function MaskDcY(delta) {
  //PROFILER_FUNC;
  var extmul = 0.00565175099786;
  var extoff = 59.04237604;
  var offset = 0.0527942789965;
  var scaler = 7.2478540673;
  var mul = 22.7326511523;
  var lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut, lut.length, delta);
}

//1614
function DiffPrecompute(xyb0, xyb1) {
  //PROFILER_FUNC;
  var xsize = xyb0.xsize();
  var ysize = xyb0.ysize();
  var result=new ImageF();result._2(xsize, ysize);
  var x2, y2;
  for (var y = 0; y < ysize; ++y) {
    if (y + 1 < ysize) {
      y2 = y + 1;
    } else if (y > 0) {
      y2 = y - 1;
    } else {
      y2 = y;
    }
    var row0_in = xyb0.Row(y);var row0_in_off = xyb0.Row_off(y);
    var row1_in = xyb1.Row(y);var row1_in_off = xyb1.Row_off(y);
    var row0_in2 = xyb0.Row(y2);var row0_in2_off = xyb0.Row_off(y2);
    var row1_in2 = xyb1.Row(y2);var row1_in2_off = xyb1.Row_off(y2);
    var row_out = result.Row(y);var row_out_off = result.Row_off(y);
    for (var x = 0; x < xsize; ++x) {
      if (x + 1 < xsize) {
        x2 = x + 1;
      } else if (x > 0) {
        x2 = x - 1;
      } else {
        x2 = x;
      }
      var sup0 = (fabs(row0_in[row0_in_off+x] - row0_in[row0_in_off+x2]) +
                     fabs(row0_in[row0_in_off+x] - row0_in2[row0_in2_off+x]));
      var sup1 = (fabs(row1_in[row1_in_off+x] - row1_in[row1_in_off+x2]) +
                     fabs(row1_in[row1_in_off+x] - row1_in2[row1_in2_off+x]));
      var mul0 = 0.975265057546;
      row_out[row_out_off+x] = mul0 * Math.min(sup0, sup1);
      var cutoff = 122.088759397;
      if (row_out[row_out_off+x] >= cutoff) {
        row_out[row_out_off+x] = cutoff;
      }
    }
  }
  return result;
}



//1656
function Mask(xyb0,
          xyb1,
          mask,
          mask_dc) {
  //PROFILER_FUNC;
  var xsize = xyb0[0].xsize();
  var ysize = xyb0[0].ysize();
  mask.length=(3);//resize
  mask_dc[0] = CreatePlanes(xsize, ysize, 3);
  var muls = [//[4]
    0.05,
    0.125412266532,
    0.231880902493,
    0.881236099962
  ];
  var normalizer = [//[2]
    1.0 / (muls[0] + muls[1]),
    1.0 / (muls[2] + muls[3])
  ];
  var r0 = 2.393872;
  var r1 = 7.224576;
  for (var i = 0; i < 2; ++i) {
    (mask)[i] = new ImageF();(mask)[i]._2(xsize, ysize);//*
    var diff = DiffPrecompute(xyb0[i], xyb1[i]);
    var blurred1 = Blur(diff, r0, 0.0);
    var blurred2 = Blur(diff, r1, 0.0);
    for (var y = 0; y < ysize; ++y) {
      for (var x = 0; x < xsize; ++x) {
        var val = normalizer[i] * (
            muls[2 * i + 0] * blurred1.Row(y)[blurred1.Row_off(y)+x] +
            muls[2 * i + 1] * blurred2.Row(y)[blurred2.Row_off(y)+x]);
        (mask)[i].Row(y)[mask[i].Row_off(y)+x] = val;
      }
    }
  }
  (mask)[2] = new ImageF();(mask)[2]._2(xsize, ysize);
  var mul = [//[2]
    12.5378252408,
    2.31907764902
  ];
  var w00 = 7.32559473935;
  var w11 = 2.66005251922;
  var w_ytob_hf = 0.55139276475;
  var w_ytob_lf = 8.95249364992;
  var p1_to_p0 = 0.0164295294749;
assert(1==1);
  for (var y = 0; y < ysize; ++y) {
    for (var x = 0; x < xsize; ++x) {
      var s0 = (mask)[0].Row(y)[mask[0].Row_off(y)+x];
      var s1 = (mask)[1].Row(y)[mask[1].Row_off(y)+x];
      var p1 = mul[1] * w11 * s1;
      var p0 = mul[0] * w00 * s0 + p1_to_p0 * p1;

      (mask)[0].Row(y)[mask[0].Row_off(y)+x] = MaskX(p0);
      (mask)[1].Row(y)[mask[1].Row_off(y)+x] = MaskY(p1);
      (mask)[2].Row(y)[mask[2].Row_off(y)+x] = w_ytob_hf * MaskY(p1);
      (mask_dc[0])[0].Row(y)[mask_dc[0][0].Row_off(y)+x] = MaskDcX(p0);
      (mask_dc[0])[1].Row(y)[mask_dc[0][1].Row_off(y)+x] = MaskDcY(p1);
      (mask_dc[0])[2].Row(y)[mask_dc[0][2].Row_off(y)+x] = w_ytob_lf * MaskDcY(p1);
    }
  }
/*if(0) {
  PrintStatistics("mask", mask);//*
  PrintStatistics("mask_dc", mask_dc[0]);//*
}*/
}

//1723
function ButteraugliDiffmap(rgb0_image,
                        rgb1_image,
                        result_image) {
  var xsize = rgb0_image[0].xsize();  
  var ysize = rgb0_image[0].ysize();  
  var kMax = 8;  if (xsize < kMax || ysize < kMax) {
     // Butteraugli values for small (where xsize or ysize is smaller    
	 // than 8 pixels) images are non-sensical, but most likely it is    
	 // less disruptive to try to compute something than just give up.    
	 // Temporarily extend the borders of the image to fit 8 x 8 size.    
	 var xborder = xsize < kMax ? ((kMax - xsize) / 2)|0 : 0;    
	 var yborder = ysize < kMax ? ((kMax - ysize) / 2)|0 : 0;    
	 var xscaled = Math.max(kMax, xsize);    
	 var yscaled = Math.max(kMax, ysize);    
	 var scaled0 = CreatePlanes(xscaled, yscaled, 3);    
	 var scaled1 = CreatePlanes(xscaled, yscaled, 3);    
	 for (var i = 0; i < 3; ++i) {
       for (var y = 0; y < yscaled; ++y) {
         for (var x = 0; x < xscaled; ++x) {
           var x2 = Math.min(xsize - 1, Math.max(0, x - xborder));          
		   var y2 = Math.min(ysize - 1, Math.max(0, y - yborder));          
		   scaled0[i].Row(y)[x] = rgb0_image[i].Row(y2)[x2];          
		   scaled1[i].Row(y)[x] = rgb1_image[i].Row(y2)[x2];        
		 }      
	   }    
	 }    
	 var diffmap_scaled=new ImageF();    
	 ButteraugliDiffmap(scaled0, scaled1, diffmap_scaled);    
	 result_image[0] = new ImageF();result_image[0]._2(xsize, ysize);    
	 for (var y = 0; y < ysize; ++y) {
      for (var x = 0; x < xsize; ++x) {
        result_image[0].Row(y)[x] = diffmap_scaled.Row(y + yborder)[x + xborder];
      }
    }
    return;
  }  
  var butteraugli= new ButteraugliComparator(rgb0_image);  
  butteraugli.Diffmap(rgb1_image, result_image);
}
//1764
function ButteraugliInterface(rgb0,
                          rgb1,
                          diffmap,
                          diffvalue) {
  var xsize = rgb0[0].xsize();  
  var ysize = rgb0[0].ysize();  
  if (xsize < 1 || ysize < 1) {
    return false;  // No image.
  }  
  for (var i = 1; i < 3; i++) {
    if (rgb0[i].xsize() != xsize || rgb0[i].ysize() != ysize ||
        rgb1[i].xsize() != xsize || rgb1[i].ysize() != ysize) {
      return false;  // Image planes must have same dimensions.
    }
  }
  ButteraugliDiffmap(rgb0, rgb1, diffmap);
  diffvalue[0] = ButteraugliScoreFromDiffmap(diffmap[0]);
  return true;
}

//1807
function ButteraugliFuzzyClass(score) {
  var fuzzy_width_up = 7.59837716738;
  var fuzzy_width_down = 5.84499454872;
  var m0 = 2.0;
  var scaler = 0.791158566821;
  var val;
  if (score < 1.0) {
    // val in [scaler .. 2.0]
    val = m0 / (1.0 + Math.exp((score - 1.0) * fuzzy_width_down));
    val -= 1.0;  // from [1 .. 2] to [0 .. 1]
    val *= 2.0 - scaler;  // from [0 .. 1] to [0 .. 2.0 - scaler]
    val += scaler;  // from [0 .. 2.0 - scaler] to [scaler .. 2.0]
  } else {
    // val in [0 .. scaler]
    val = m0 / (1.0 + Math.exp((score - 1.0) * fuzzy_width_up));
    val *= scaler;
  }
  return val;
}

//1827
function ButteraugliFuzzyInverse(seek) {
  var pos = 0;
  for (var range = 1.0; range >= 1e-10; range *= 0.5) {
    var cur = ButteraugliFuzzyClass(pos);
    if (cur < seek) {
      pos -= range;
    } else {
      pos += range;
    }
  }
  return pos;
}

