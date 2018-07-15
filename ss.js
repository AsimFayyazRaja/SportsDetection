var screenshot = require('desktop-screenshot');

const fs = require('fs');

//takes screenshot, and names the SS as num of files in the folder
function take_screenshot(){
    var dir = './test';
    var tosave="/test/"
    var basepath="./currdir"		//add current dir here
    var path=basepath+tosave

    fs.readdir(dir, (err, files) => {       //getting num of files in folder to name SS
        //console.log(files.length);
        var i=files.length;
        //console.log(path)
        screenshot(path+"testimage"+i+".png", function(error, complete) {
            if(error)
                console.log("Screenshot failed", error);
            else
                console.log("Screenshot succeeded");
        });
      });
}

delay=1000
setInterval(take_screenshot, delay)     //calls the func after some delay   

