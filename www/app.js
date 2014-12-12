var express = require('express');
var app = express();

app.get('/test', function (req, res) {
	if(req.query.input_text){
		console.log('At python code w/ text: ' + req.query.input_text)
		var args = [__dirname + '/public/predict_rotd.py',
		req.query.input_text,
		__dirname + '/public/sgdc_pipe.p'
		];
		var python = require('child_process').spawn(
		'python', args);
		console.log(args[0] +' '+ args[1] +' '+ args[2])
		var output = '';
		python.stdout.on('data', function(){ output += data });
		python.on('close', function(code){
			console.log('python finished: ' + output)
			if (code !== 0) {
				 return res.status(500).send(String(code)); 
			}
			return res.status(200).send(output);
		})
	} else {  res.status(200).send('No Input') }

	//res.send('Got the GET: ' + req.query.input_text)
})

app.use(express.static(__dirname + '/public'))

var server = app.listen(8080, function(){
		
	var host = server.address().address
	var port = server.address().port

	console.log('Example app listening at http://%s:%s', host, port)

})
