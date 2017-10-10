# butteraugli.js
hand port of google butteraugli in javascript

Demo: http://libwebpjs.hohenlimburg.org/butteraugli/

## Installation

```
npm install butteraugli
```

## Usage

### Description

```js
var butteraugli = require("butteraugli")
var headmap = [] /* return values as RGB Array (optional feature)*/
butteraugli({data:[0,0,0,255],width:0,height:0},{data:[0,0,0,255],width:0,height:0},headmap)
/*description 2x inputs*/
{
	data:[0,0,0,255/*, ... */], /* currently only rgba values allowed */
	width:0, /* image width */
	height:0 /* image height */
}
```


### Example

```js
getPixels("example1.png", function(err, pixels1) {
  if(err) {
    console.log("Bad image path")
    return
  }

  getPixels("example2.png", function(err, pixels2) {
    if(err) {
      console.log("Bad image path")
      return
    }

  var headmap = [] /* return values in RGB Array (optional feature)*/
  var input1 = {data:pixels1.data,width:pixels1.shape[0],height:pixels1.shape[1]}
  var input2 = {data:pixels2.data,width:pixels2.shape[0],height:pixels2.shape[1]}
  var score = butteraugli(input1,input2,headmap)
  console.log(score)
  })
})
```