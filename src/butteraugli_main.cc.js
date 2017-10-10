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
// Disclaimer: This is not an official Google product.
//
// Author: Jyrki Alakuijala (jyrki.alakuijala@gmail.com)
//         Dominik Homberger (dominik.homberger@gmail.com)

function NewSrgbToLinearTable() {
  var table = [];//256
  for (var i = 0; i < 256; ++i) {
    var srgb = i / 255.0;
    table[i] =
        255.0 * (srgb <= 0.04045 ? srgb / 12.92
                                 : Math.pow((srgb + 0.055) / 1.055, 2.4));
  }
  return table;
}

// Translate R, G, B channels from sRGB to linear space. If an alpha channel
// is present, overlay the image over a black or white background. Overlaying
// is done in the sRGB space; while technically incorrect, this is aligned with
// many other software (web browsers, WebP near lossless).
function FromSrgbToLinear(rgb,
                      linear, background) {
  var xsize = rgb[0].xsize();
  var ysize = rgb[0].ysize();
  var kSrgbToLinearTable = NewSrgbToLinearTable();

  if (rgb.length == 3) {  // RGB
    for (var c = 0; c < 3; c++) {
		var image_=new ImageF();image_._2(xsize, ysize);
      linear.push(image_);
      for (var y = 0; y < ysize; ++y) {
        var row_rgb = rgb[c].Row(y);var row_rgb_off = rgb[c].Row_off(y);
        var row_linear = linear[c].Row(y);var row_linear_off = linear[c].Row_off(y);
        for (var x = 0; x < xsize; x++) {
          var value = row_rgb[row_rgb_off+x];
          row_linear[row_linear_off+x] = kSrgbToLinearTable[value];
        }
      }
    }
  } else {  // RGBA
    for (var c = 0; c < 3; c++) {
		var image_=new ImageF();image_._2(xsize, ysize);
      linear.push(image_);
      for (var y = 0; y < ysize; ++y) {
        var row_rgb = rgb[c].Row(y);var row_rgb_off = rgb[c].Row_off(y);
        var row_linear = linear[c].Row(y);var row_linear_off = linear[c].Row_off(y);
        var row_alpha = rgb[3].Row(y);var row_alpha_off = rgb[3].Row_off(y);
        for (var x = 0; x < xsize; x++) {
          var value;
          if (row_alpha[row_alpha_off+x] == 255) {
            value = row_rgb[row_rgb_off+x];
          } else if (row_alpha[row_alpha_off+x] == 0) {
            value = background;
          } else {
            var fg_weight = row_alpha[row_alpha_off+x];
            var bg_weight = 255 - fg_weight;
            value =
                ((row_rgb[row_rgb_off+x] * fg_weight + background * bg_weight + 127) / 255)|0;
          }
          row_linear[row_linear_off+x] = kSrgbToLinearTable[value];
        }
      }
    }
  }
}

//312
function ScoreToRgb(score, score_off, good_threshold,
                    bad_threshold, rgb, rgb_off) {
	var score=score[score_off];
  var heatmap = [//[12][3]
    [ 0, 0, 0 ],
    [ 0, 0, 1 ],
    [ 0, 1, 1 ],
    [ 0, 1, 0 ], // Good level
    [ 1, 1, 0 ],
    [ 1, 0, 0 ], // Bad level
    [ 1, 0, 1 ],
    [ 0.5, 0.5, 1.0 ],
    [ 1.0, 0.5, 0.5 ],  // Pastel colors for the very bad quality range.
    [ 1.0, 1.0, 0.5 ],
    [ 1, 1, 1, ],
    [ 1, 1, 1, ]
  ];
  if (score < good_threshold) {
    score = (score / good_threshold) * 0.3;
  } else if (score < bad_threshold) {
    score = 0.3 + (score - good_threshold) /
        (bad_threshold - good_threshold) * 0.15;
  } else {
    score = 0.45 + (score - bad_threshold) /
        (bad_threshold * 12) * 0.5;
  }
  var kTableSize = heatmap.length;// sizeof(heatmap) / sizeof(heatmap[0]);
  score = Math.min(Math.max(
      score * (kTableSize - 1), 0.0), kTableSize - 2);
  var ix = (score)|0;//static_cast<int>
  var mix = score - ix;
  for (var i = 0; i < 3; ++i) {
    var v = mix * heatmap[ix + 1][i] + (1 - mix) * heatmap[ix][i];
    rgb[rgb_off+i] = (255 * Math.pow(v, 0.5) + 0.5)|0;//static_cast<uint8_t>
  }
}

//348
function CreateHeatMapImage(distmap, good_threshold,
                        bad_threshold, xsize, ysize,
                        heatmap) {
  heatmap.length=(3 * xsize * ysize);//resize
  for (var y = 0; y < ysize; ++y) {
    for (var x = 0; x < xsize; ++x) {
      var px = xsize * y + x;
      var d = distmap.Row(y);var d_off = distmap.Row_off(y)+x;
      var rgb = (heatmap);var rgb_off = 3 * px;//* &*
      ScoreToRgb(d, d_off, good_threshold, bad_threshold, rgb, rgb_off);
    }
  }
}

function readRGBA(rgba) {
	var xsize=rgba.width;
	var ysize=rgba.height;
  var rgb = [];// CreatePlanes(xsize, ysize, 3);
  for (var i = 0; i < 3; ++i) {
	  var image8=new Image8();
	  image8._2(xsize, ysize);
	  rgb.push(image8);
  }
  for (var y = 0; y < ysize; ++y) {
	var row = rgba.data; var row_off=xsize*4*y;
	var row0 = rgb[0].Row(y);var row0_off = rgb[0].Row_off(y);
	var row1 = rgb[1].Row(y);var row1_off = rgb[1].Row_off(y);
	var row2 = rgb[2].Row(y);var row2_off = rgb[2].Row_off(y);
	//var row3 = rgb[3].Row(y);var row3_off = rgb[3].Row_off(y);

	for (var x = 0; x < xsize; ++x) {
	  row0[row0_off+x] = row[row_off+4 * x + 0];
	  row1[row1_off+x] = row[row_off+4 * x + 1];
	  row2[row2_off+x] = row[row_off+4 * x + 2];
	  //row3[row3_off+x] = row[row_off+4 * x + 3];
	}
  }
  return rgb;
}

function butteraugli(rgba1,rgba2,headmap) {
var rgb1= readRGBA(rgba1);
var rgb2= readRGBA(rgba2);
    if (rgb1.width != rgb2.width ||
        rgb1.height != rgb2.height) {
			console.log("The images are not equal in size: (%lu,%lu) vs (%lu,%lu)");
      /*fprintf(
          stderr, "The images are not equal in size: (%lu,%lu) vs (%lu,%lu)\n",
          rgb1[c].xsize(), rgb2[c].xsize(), rgb1[c].ysize(), rgb2[c].ysize());*/
      return 1;
  }

  var linear1=[], linear2=[];//std::vector<ImageF>

  FromSrgbToLinear(rgb1, linear1, 0);
  FromSrgbToLinear(rgb2, linear2, 0);
  var diff_map=[new ImageF()], diff_map_on_white=new ImageF();
  var diff_value=[];//double
  if (!ButteraugliInterface(linear1, linear2, diff_map,
                                         diff_value)) {
    console.log("Butteraugli comparison failed\n");
    return 1;
  }
  diff_value=diff_value[0];
  var diff_map_ptr = diff_map[0];//ImageF* &
  if (rgb1.size == 4 || rgb2.size == 4) {
    // If the alpha channel is present, overlay the image over a white
    // background as well.
    FromSrgbToLinear(rgb1, linear1, 255);
    FromSrgbToLinear(rgb2, linear2, 255);
    var diff_value_on_white=[];//double
    if (!ButteraugliInterface(linear1, linear2, diff_map_on_white,
                                           diff_value_on_white)) {
      console.log("Butteraugli comparison failed\n");
      return 1;
    }
	diff_value_on_white=diff_value_on_white[0];
    if (diff_value_on_white > diff_value) {
      diff_value = diff_value_on_white;
      diff_map_ptr = diff_map_on_white;//&
    }
  }
  console.log(diff_value);//%lf

  if (headmap) {
    var good_quality = ButteraugliFuzzyInverse(1.5);
    var bad_quality = ButteraugliFuzzyInverse(0.5);
    var rgb=headmap;//[];//std::vector<uint8_t>
    CreateHeatMapImage(diff_map_ptr, good_quality, bad_quality,//*
                       rgb1[0].xsize(), rgb2[0].ysize(), rgb);
	//return rgb;
  }

  return diff_value;
  return 0;
}
if(typeof window !== 'undefined') window['butteraugli']=butteraugli;