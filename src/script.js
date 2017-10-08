function EventListener(obj,evt,fnc,useCapture){
	if (!useCapture) useCapture=false;
	if (obj.addEventListener){
		obj.addEventListener(evt,fnc,useCapture);
		return true;
	} else if (obj.attachEvent) return obj.attachEvent("on"+evt,fnc);
	else{
		MyAttachEvent(obj,evt,fnc);
		obj['on'+evt]=function(){ MyFireEvent(obj,evt) };
	}
} 

function MyAttachEvent(obj,evt,fnc){
	if (!obj.myEvents) obj.myEvents={};
	if (!obj.myEvents[evt]) obj.myEvents[evt]=[];
	var evts = obj.myEvents[evt];
	evts[evts.length]=fnc;
}
function MyFireEvent(obj,evt){
	if (!obj || !obj.myEvents || !obj.myEvents[evt]) return;
	var evts = obj.myEvents[evt];
	for (var i=0,len=evts.length;i<len;i++) evts[i]();
}

window.onload=function() {
	//var ba=new butteraugliLib();
var image=[],imageContext=[],img=[];
var butteraugliExample=document.getElementById('example');var butteraugliBtn=document.getElementById('butteraugli');var score=document.getElementById('score');
var headmapcanvas=document.getElementById('headmap');var headmapContext=headmapcanvas.getContext("2d");
for(var i=1;i<3;++i) {
	image[i] = document.getElementById("image"+i), imageContext[i] = image[i].getContext("2d");
	image[i].nr = i;

	img[i] = document.createElement("img");
	img[i].nr = i;

	if (typeof FileReader !== "undefined")	
	imageContext[i].fillText("Drop an image here and press the \"encoding\" button ", 25, 80);
	else
	imageContext[i].fillText("Choise an sample image", 86, 80);
		
	EventListener(img[i],"load", function (evt) {
		
		var i = evt.currentTarget.nr;
		imageContext[i].clearRect(0, 0, image[i].width, image[i].height);
		image[i].width=img[i].width;
		image[i].height=img[i].height;
		imageContext[i].drawImage(img[i], 0, 0);
		score.disabled=false;
	}, false);
	EventListener(image[i],"dragenter", function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
	}, false);

	EventListener(image[i],"dragover", function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
	}, false);

	EventListener(image[i],"drop", function (evt) {
		var i = evt.currentTarget.nr;
			evt.preventDefault();
			evt.stopPropagation();
		
		var files = evt.dataTransfer.files;
		if (files.length > 0) {
			var file = files[0];
			if (typeof FileReader !== "undefined") {
				if (file.type.indexOf("image") != -1) {
					var freader = new FileReader();
					freader.onload = function (evt) {
						img[i].src=evt.target.result;
					};
					freader.readAsDataURL(file);
				} else {
					alert('Your Browser don\'t support the Filereader API');
				}
			}
		}
	}, false);
}
	EventListener(butteraugliBtn,"click", function (evt) {
		var rgbadata1=imageContext[1].getImageData(0,0,image[1].width,image[1].height);
		var rgbadata2=imageContext[2].getImageData(0,0,image[2].width,image[2].height);
		headmapcanvas.width=image[1].width;
		headmapcanvas.height=image[1].height;
		var headmapdata=headmapContext.getImageData(0,0,headmapcanvas.width,headmapcanvas.height);
		var headmapdata3bytes=[];
		butteraugliBtn.value='Butteraugli (waiting)';
		butteraugliBtn.disabled=true;
		var start = new Date();
		score.innerHTML=butteraugli(rgbadata1,rgbadata2,headmapdata3bytes);
		score.innerHTML+=' (Speed: '+(new Date()-start)+' ms)';
		
		for(var i=0,a=0;i<headmapdata3bytes.length;a+=4,i+=3) {
			headmapdata.data[a+0]=headmapdata3bytes[i+0];
			headmapdata.data[a+1]=headmapdata3bytes[i+1];
			headmapdata.data[a+2]=headmapdata3bytes[i+2];
			headmapdata.data[a+3]=255;
		}
		headmapContext.putImageData(headmapdata,0,0);
		butteraugliBtn.value='Butteraugli';
		butteraugliBtn.disabled=false;
	}, false);
	EventListener(butteraugliExample,"click", function (evt) {
		img[1].src='test_images/1.png';
		img[2].src='test_images/out.png';
	}, false);
}