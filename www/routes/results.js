var express = require('express');
var router = express.Router();
var fs = require('fs');
var _ = require('underscore');

router.post('/results', function (req, res) {
    var data = req.body;
    var path = "./public/8214ba9ab15f49/test.dat";
    if (!fs.existsSync(path)) {
        var s = _.chain(Object.getOwnPropertyNames(data))
            .reduce(function (s1, s2) { return s1 + ", " + s2; })
            .value();
        fs.appendFileSync(path, s + "\n");
    }
    var s = _.chain(Object.getOwnPropertyNames(data))
        .map(function (prop) { return data[prop].toString(); })
        .reduce(function (s1, s2) { return s1 + ", " + s2; })
        .value();

    fs.appendFile(path, s + "\n");
    res.sendStatus(200);
});

router.post('/survey', function (req, res) {
    var data = req.body;
    var path = "./public/8214ba9ab15f49/survey.dat";
    var fields = [
        'id',
        'literacy',
        'knowledge',
        'experience',
        'professional',
        'years',
        'role',
        'personal',
        '401k',
        'investType',
        'stocks',
        'bonds',
        'cash',
        'other',
        'return'];

    if (!fs.existsSync(path)) {
        var s = _.chain(fields)
            .reduce(function (s1, s2) { return s1 + ", " + s2; })
            .value();
        fs.appendFileSync(path, s + "\n");
    }
    var s = _.chain(fields)
        .map(function (prop) {
            return data.hasOwnProperty(prop) ? data[prop].toString() : 'NA';
        })
        .reduce(function (s1, s2) { return s1 + ", " + s2; })
        .value();

    fs.appendFile(path, s + "\n");

    var path = "./public/8214ba9ab15f49/users.dat";
    var s = data.id + '\n'
    fs.appendFile(path, s);

    res.sendStatus(200);
});



module.exports = router;
