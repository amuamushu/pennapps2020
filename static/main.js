let video;
function stop () { 
    var stream = video.srcObject; 
    var tracks = stream.getTracks(); 
    for (var i = 0; i < tracks.length; i++) { 
        var track = tracks[i]; 
        track.stop(); 
    } 
    video.srcObject = null; 
} 

function start () { 
    video = document.getElementById('webcam'), 
        vendorUrl = window.URL || window.webkitURL; 
    if (navigator.mediaDevices.getUserMedia) { 
        navigator.mediaDevices.getUserMedia({ video: true }) 
            .then(function (stream) { 
                video.srcObject = stream; 
            }).catch(function (error) { 
                console.log("Something went wrong!"); 
            }); 
    } 
} 

// Get handles on the video and canvas elements

let body = document.querySelector('body');
let canvas = document.createElement('canvas');
body.appendChild(canvas);
// Get a handle on the 2d context of the canvas element
var context = canvas.getContext('2d');
// Define some vars required later
var w, h, ratio;
let mp4 = document.getElementById('video')
// Add a listener to wait for the 'loadedmetadata' state so the video's dimensions can be read
mp4.addEventListener('loadedmetadata', function() {
  // Calculate the ratio of the video's width to height
  ratio = mp4.videoWidth / mp4.videoHeight;
  // Define the required width as 100 pixels smaller than the actual video's width
  w = mp4.videoWidth - 100;
  // Calculate the height based on the video's width and the ratio
  h = parseInt(w / ratio, 10);
  // Set the canvas width and height to the values just calculated
  canvas.width = w;
  canvas.height = h;			
}, false);

// Takes a snapshot of the video
function snap() {
  // Define the size of the rectangle that will be filled (basically the entire element)
  context.fillRect(0, 0, w, h);
  // Grab the image from the video
  context.drawImage(mp4, 0, 0, w, h);
}