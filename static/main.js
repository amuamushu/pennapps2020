
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
    var video = document.getElementById('video'), 
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