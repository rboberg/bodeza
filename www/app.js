var express = require('express');
var app = express();

app.get('/test', function (req, res) {
	if(req.query.input_text){
		console.log('At python code w/ text: ' + req.query.input_text)
		var args = [__dirname + '/public/predict_rotd.py',
		'"' + req.query.input_text + '"',
		__dirname + '/public/sgdc_pipe.p'
		];
		var cmd = 'python'
		for(i=0;i<args.length;i++){
			cmd += (' ' + args[i])
		}
		cmd = cmd.replace(/"/g, "");
		console.log(cmd);
		var child;
		child = require('child_process').exec(cmd,
			function(error, stdout, stderr){
				console.log('stdout: ' + stdout);
				res.send(stdout);
				console.log('stderr: ' + stderr);
				if (error !== null) {
					console.log('exec error: ' + error);
				}
			});
	} else {  res.status(200).send('No Input') }

})

app.use(express.static(__dirname + '/public'))

var server = app.listen(8080, function(){
		
	var host = server.address().address
	var port = server.address().port

	console.log('Example app listening at http://%s:%s', host, port)

})
