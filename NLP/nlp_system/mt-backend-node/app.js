var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var bodyParser = require('body-parser');
var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');
var http = require('http');
var https = require('https');
var app = express();
var fs = require('fs');

CONFIG = require('./config');
// 设置handlebars为默认模板引擎
let handlebars = require('express-handlebars').create(require('./lib/handlebarsConfig'));
app.engine('handlebars', handlebars.engine);
app.set('view engine', 'handlebars');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));
//  引用body-parser，使得可以获取网页的POST请求
app.use(bodyParser.json({limit: '50mb'}));
app.use(bodyParser.urlencoded({limit: '50mb', extended: true}));

app.use('/upload',express.static(__dirname + '/upload'))
// app.get('/s', indexRouter.index);
// app.get('/', indexRouter.index);

// app.get('/getimg', indexRouter.getimg)

// app.get('/sousuo',indexRouter.sousuo)
// app.post('/chart',indexRouter.chart)
// app.post('/weblogin_wait',indexRouter.weblogin_wait)
// app.post('/weblogin_result',indexRouter.weblogin_result)

// app.get('/webwantkey',usersRouter.webwantkey)
// app.post('/ajax_webkey',usersRouter.ajax_webkey)

// app.post('/updateqrcode',usersRouter.updateqrcode)

app.get('/test',function(req,res){
    res.send("test");
})

app.post('/login',indexRouter.login)

app.post('/uploadwenzi',usersRouter.uploadwenzi)

app.get('/emoji_predict',usersRouter.emoji_predict)
app.get('/ner',usersRouter.ner)
app.get('/translate',usersRouter.translate)
app.get('/search',usersRouter.search)
app.get('/get_random_sentence',usersRouter.get_random_sentence)

app.post('/uploadmusicalnotes',usersRouter.uploadmusicalnotes)

app.post('/uploadimg',usersRouter.uploadimg)

app.post('/uploadvideo',usersRouter.uploadvideo)

app.post('/getrank',indexRouter.getrank)

app.post('/getmydynamic',usersRouter.getmydynamic)

app.post('/deletemydynamic',usersRouter.deletemydynamic)

app.post('/deletemyaudio',usersRouter.deletemyaudio)

app.post('/gettimelinedynamic',usersRouter.gettimelinedynamic)

app.post('/dy_zan',usersRouter.dy_zan)

app.post('/getmyzan',usersRouter.getmyzan)
// app.post('/resign',usersRouter.resign)
// app.post('/loginsercret',usersRouter.loginsercret)
// app.get('/loginpage',usersRouter.loginpage)


// app.get('/myspace',usersRouter.myspace)
// app.post('/getnicenews',usersRouter.getnicenews)

// app.post('/addnew',usersRouter.addnew)
// app.post('/addsc',usersRouter.addsc)
//app.post('/getuserinfo',usersRouter.getuserinfo)

app.use(function (req, res) {
    res.status(404);
    res.render('404',{
    	layout: null
    });
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error',{
    	layout: null
    });
});

// let port = 8081
// app.listen(port,function(){
// 	console.log('express is listening on the port ' + port)
// })
// let port = 80
const httpsOption = {
  key:fs.readFileSync("./cert/2_ai-music.xyz.key"),
  cert:fs.readFileSync("./cert/1_ai-music.xyz_bundle.crt")
}
// app.listen(port, function () {
//     console.log('Express started on http://localhost:' + port + '; press Ctrl-C to terminate.');
// });
http.createServer(app).listen(8889,()=>{
    console.log('Express started on http://localhost:' + 8889 + '; press Ctrl-C to terminate.')
})
// https.createServer(httpsOption, app).listen(8889,()=>{
//     console.log('Express started on https://localhost:' + 8889 + '; press Ctrl-C to terminate.')
// })

module.exports = app;


