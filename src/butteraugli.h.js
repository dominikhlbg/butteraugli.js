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

var Image8 = Image_;
var ImageF = Image_;

function AllocateArray(length) {
	return mallocArr(length,0);
}

function Image_(T) {
	var T=1;
  this.xsize_=0;
  this.ysize_=0;
  this.bytes_per_row_=0;
  this.bytes_=null;
  this._2=function(xsize, ysize) {
    this.xsize_=(xsize),
    this.ysize_=(ysize),
    this.bytes_per_row_=(this.BytesPerRow(xsize)),
    this.bytes_=(AllocateArray(this.bytes_per_row_ * ysize));
  }
  this._3=function(xsize, ysize, val) {
    this.xsize_=(xsize),
    this.ysize_=(ysize),
    this.bytes_per_row_=(this.BytesPerRow(xsize)),
    this.bytes_=(AllocateArray(this.bytes_per_row_ * ysize));
    for (var y = 0; y < this.ysize_; ++y) {
      var row = this.Row(y);
	  var row_off = this.Row_off(y);
      for (var x = 0; x < this.xsize_; ++x) {
        row[row_off+x] = val;
      }
      // Avoid use of uninitialized values msan error in padding in WriteImage
      memset(row, row_off + this.xsize_, 0, this.bytes_per_row_ - T * this.xsize_);//sizeof(T)
    }
  }

  // Useful for pre-allocating image with some padding for alignment purposes
  // and later reporting the actual valid dimensions.
  this.ShrinkTo=function(xsize, ysize) {
    assert(xsize < this.xsize_);
    assert(ysize < this.ysize_);
    this.xsize_ = xsize;
    this.ysize_ = ysize;
  }

  // How many pixels.
  this.xsize=function() { return this.xsize_; }
  this.ysize=function() { return this.ysize_; }

  // Returns pointer to the start of a row, with at least xsize (rounded up to
  // the number of vector lanes) accessible values.
  this.Row=function(y) {
    //assert(y < this.ysize_);
    var row = this.bytes_;// + y * this.bytes_per_row_
    return row;//(PIK_ASSUME_ALIGNED(row, 64));
  }
  this.Row_off=function(y) {
    //assert(y < this.ysize_);
    var row_off = + y * this.bytes_per_row_;//this.bytes_ 
    return row_off;//(PIK_ASSUME_ALIGNED(row, 64));
  }
  this.bytes_per_row=function() { return this.bytes_per_row_; }

  // Returns cache-aligned row stride, being careful to avoid 2K aliasing.
  this.BytesPerRow=function(xsize) {assert(xsize>0);
    // lowpass reads one extra AVX-2 vector on the right margin.
    var row_size = xsize * T + 32;//sizeof(T)
    var align = 64;// (new CacheAligned()).kCacheLineSize;
    var bytes_per_row = (row_size + align - 1) & ~(align - 1);
    // During the lengthy window before writes are committed to memory, CPUs
    // guard against read after write hazards by checking the address, but
    // only the lower 11 bits. We avoid a false dependency between writes to
    // consecutive rows by ensuring their sizes are not multiples of 2 KiB.
    if (bytes_per_row % 2048 == 0) {
      bytes_per_row += align;
    }
    return bytes_per_row;
  }

}


function PsychoImage() {
	
  this.uhf=[];
  this.hf=[];
  this.mf=[];
  this.lf=[];
};


/*function ButteraugliComparator() {
  this.xsize_=0;
  this.ysize_=0;
  this.num_pixels_=0;
  this.pi0_=new PsychoImage();
}*/

// Returns newly allocated planes of the given dimensions.
//316
//template <typename T>
function CreatePlanes(xsize,
                      ysize,
                      num_planes) {
  var planes=mallocArrOI(num_planes,Image_);//std::vector<Image<T>>
  //planes.reserve(num_planes);
  for (var i = 0; i < num_planes; ++i) {
    planes[i]._2(xsize, ysize);//emplace_back
  }
  return planes;
}

// Returns a new image with the same dimensions and pixel values.
//329
//template <typename T>
function CopyPixels(other) {
  var copy=new Image_();copy._2(other.xsize(), other.ysize());
  var from = other.bytes_;//();
  var to = copy.bytes_;//();
  memcpy(to,0, from,0, other.ysize() * other.bytes_per_row());
  return copy;
}

//472
function RgbToXyb(r, g, b,
                  valx,valx_off,
                  valy,valy_off,
                  valb,valb_off) {
  valx[valx_off+0] = r - g;
  valy[valy_off+0] = r + g;
  valb[valb_off+0] = b;
}

//482
function OpsinAbsorbance(in0, in1,
                         in2,
                         out0,
                         out1,
                         out2) {
  // https://en.wikipedia.org/wiki/Photopsin absorbance modeling.
  var mixi0 = 0.262805861774;
  var mixi1 = 0.447726163795;
  var mixi2 = 0.0669350599301;
  var mixi3 = 0.70582780208;
  var mixi4 = 0.242970172936;
  var mixi5 = 0.557086443066;
  var mixi6 = mixi2;
  var mixi7 = mixi3;
  var mixi8 = 0.443262270088;
  var mixi9 = 1.22484933589;
  var mixi10 = 0.610100334382;
  var mixi11 = 5.95035078154;

  var mix0=(mixi0);
  var mix1=(mixi1);
  var mix2=(mixi2);
  var mix3=(mixi3);
  var mix4=(mixi4);
  var mix5=(mixi5);
  var mix6=(mixi6);
  var mix7=(mixi7);
  var mix8=(mixi8);
  var mix9=(mixi9);
  var mix10=(mixi10);
  var mix11=(mixi11);

  out0[0] = mix0 * in0 + mix1 * in1 + mix2 * in2 + mix3;
  out1[0] = mix4 * in0 + mix5 * in1 + mix6 * in2 + mix7;
  out2[0] = mix8 * in0 + mix9 * in1 + mix10 * in2 + mix11;
}

