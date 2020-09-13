let mp4 = document.getElementById('video')

mp4.onpause = (event) => {
// TODO
  console.log('The Boolean paused property is now true. Either the ' + 
  'pause() method was called or the autoplay attribute was toggled.');
};