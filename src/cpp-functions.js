/*onmessage =  function(evt) {
var height=[0];
var width=[0];
var result = evt.data;
var response = result.response;
var start=new Date();
var rgba = WebPDecodeRGBA(response,0,response.length,width,height);
var speed = ((new Date())-start);
var transferspeed = new Date();
var data = {'rgba':rgba,'width':width,'height':height,'speed':speed,'transferspeed':transferspeed,'thread':result.thread};
postMessage(data,[data.rgba.buffer]);
};*/

function assert(bCondition) {
	if (!bCondition) {
		throw new Error('assert :P');
	}
}

function static_assert(bCondition,str) {
	if (!bCondition) {
		throw new Error(str);
	}
}

function memcmp(data, data_off, str, size) {
	for(var i=0;i<size;i++)
		if(data[data_off+i]!=str.charCodeAt(i))
			return true;
	return false;
}

function memcpy(dest,dest_off,src,src_off,num) {
/*	if(typeof src.BYTES_PER_ELEMENT==='undefined')
	dest=dest;
	dest.set(src.subarray(src_off,src_off+num),dest_off);*/
	for(var i=0;i<num;i++) {
		//if(typeof dest[dest_off+i]!=='number'||typeof src[src_off+i]!=='number')
		//dest=dest;
		//assert(typeof dest[dest_off+i]==='number');
		//assert(typeof src[src_off+i]==='number');
		dest[dest_off+i]=src[src_off+i];
	}
}

function memset(ptr,ptr_off,value,num) {
	for(var i=0;i<num;i++)
		ptr[ptr_off+i]=value;
}
function mallocArr(size,value) {
	var arr=new Float32Array(size);// Array();
	/*for(var i=0;i<size;i++)
	arr.push(value);*/
	return arr;
}

function mallocArrOI(size,value) {
	var arr=new Array();
	for(var i=0;i<size;i++)
	arr.push(new value());
	return arr;
}

function mallocMArr(size,value) {
	function buildarray(arr,sub,counts) { 
		var l=counts[sub];
		for(var i=0;i<l;i++) {
			arr.push((counts.length>sub+1)?new Array():value);
			if(counts.length<sub+1) return;
			buildarray(arr[i],sub+1,counts);
		}
	}
	var arr=new Array();
	buildarray(arr,0,size);
	return arr;
}

function mallocMArrOI(size,value) {
	function buildarray(arr,sub,counts) { 
		var l=counts[sub];
		for(var i=0;i<l;i++) {
			arr.push((counts.length>sub+1)?new Array():new value());
			if(counts.length<sub+1) return;
			buildarray(arr[i],sub+1,counts);
		}
	}
	var arr=new Array();
	buildarray(arr,0,size);
	return arr;
}
function memmove(destination, destination_off, source, source_off, num) {
	//copy from last to start
	var i; var temp=[];
	for(i=num-1;i>=0;--i) {
		temp[i]=source[source_off+i];
	}
	for(i=num-1;i>=0;--i) {
		destination[destination_off+i]=temp[i];
	}
}
function fabs(value) {
	return Math.abs(value);
}