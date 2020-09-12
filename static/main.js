
let lastPeriodIndex = -1;
let /* Array<String> */ previousKeyWords = [];

/**
 * Gets the text the user has inputted.
 * 
 * <p>This gets called every time the textbox changes
 */
function getUserText() {
  // This will be all the text the user has written so far.
  // We would like to filter it to be the most recent words.
  let userText =  document.getElementById('text-area').value;
  // Only process the text by sentences.
  if (userText[userText.length - 1] == '.') {
      let lastSentence = userText.substring(lastPeriodIndex + 1);
      lastPeriodIndex = userText.length - 1;
      // We decided that sentences less than 3 characters in length
      // are not important.
      if (lastSentence.length < 3) {
          return;
      }
      // TODO: Send a post request to a servlet in order to call the API
      // on last sentence sentence.
      console.log(lastSentence);
  }
}

// TODO: Create a function for rendering the information on the DOM
// once it is fetched from the API

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