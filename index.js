function include(filename) {
	var fs = require("fs")
	var vm = require('vm')
	
	vm.runInThisContext(fs.readFileSync(__dirname + "/" + filename))
}

var files=['cpp-functions.js','butteraugli.h.js','butteraugli.cc.js','butteraugli_main.cc.js']

for(var i in files) {
	include('src/'+files[i])
}
module.exports = butteraugli