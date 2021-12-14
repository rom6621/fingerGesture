const video = document.getElementById("video");
let src = null;
let dst = null;
let cap = null;
let model = null;

async function main() {
  tf.setBackend('webgl');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false,
  });
  video.srcObject = stream;
  video.addEventListener("canplay", readyCapture, false);
  console.log(stream);
  video.play();
}

async function readyCapture() {
  video.width = video.videoWidth;
  video.height = video.videoHeight;
  model = await handpose.load();
  processVideo();
}

function getDistance(ary1, ary2) {
  return Math.sqrt( Math.pow( ary2[0]-ary1[0], 2 ) + Math.pow( ary2[1]-ary1[1], 2 ) ) ;
}

async function processVideo() {
  const predictions = await model.estimateHands(video);
  src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
  cap = new cv.VideoCapture(video);
  cap.read(src);
  cv.cvtColor(src, dst, cv.COLOR_RGBA2RGB);

  if (predictions.length > 0) {
    const landmarks = [];
    for(let i=0; i<predictions[0].landmarks.length; i++) {
      const landmark = predictions[0].landmarks[i];
      landmarks.push((landmark[0], landmark[1]))
      cv.circle(dst, new cv.Point(landmark[0], landmark[1]), 5, new cv.Scalar(255, 255, 0), 2)
    }
    console.log(landmarks[0]);
    const f1 = getDistance(landmarks[0], landmarks[4]);
    const f2 = getDistance(landmarks[0], landmarks[8]);
    const f3 = getDistance(landmarks[0], landmarks[12]);
    const f4 = getDistance(landmarks[0], landmarks[16]);
    const f5 = getDistance(landmarks[0], landmarks[20]);
    const f6 = getDistance(landmarks[0], landmarks[3]);
    const f7 = getDistance(landmarks[0], landmarks[6]);
    const f8 = getDistance(landmarks[0], landmarks[10]);
    const f9 = getDistance(landmarks[0], landmarks[14]);
    const f10 = getDistance(landmarks[0], landmarks[18]);
    const f11 = getDistance(landmarks[4], landmarks[6]);
    if (f1>f2 && f1>f3 && f1>f4 && f1>f5) {
      if (f11>52 && landmarks[3][1]-landmarks[4][1]>0) {
        cv.putText(dst, "Good" , new cv.Point(10, 210), cv.FONT_HERSHEY_SIMPLEX, 1.0, new cv.Scalar(0, 255, 255), 2, cv.LINE_AA);
      } else if(f11>52){
        cv.putText(dst, "Bad" , new cv.Point(10, 210), cv.FONT_HERSHEY_SIMPLEX, 1.0, new cv.Scalar(0, 255, 255), 2, cv.LINE_AA);
      } else{
        cv.putText(dst, "Rock" , new cv.Point(10, 210), cv.FONT_HERSHEY_SIMPLEX, 1.0, new cv.Scalar(0, 255, 255), 2, cv.LINE_AA);
      }
    } else if(f1>f6 && f2>f7 && f3>f8 && f4>f9 && f5>f10) {
      cv.putText(dst, "Paper" , new cv.Point(10, 210), cv.FONT_HERSHEY_SIMPLEX, 1.0, new cv.Scalar(0, 255, 255), 2, cv.LINE_AA);
    } else if(f2>f7 && f3>f8 && f4<f9 && f5<f10){
      cv.putText(dst, "Scissors" , new cv.Point(10, 210), cv.FONT_HERSHEY_SIMPLEX, 1.0, new cv.Scalar(0, 255, 255), 2, cv.LINE_AA);
    } else if(f2<f7 && f3>f8 && f4<f9 && f5<f10){
      cv.putText(dst, "Fuxx" , new cv.Point(10, 210), cv.FONT_HERSHEY_SIMPLEX, 1.0, new cv.Scalar(0, 255, 255), 2, cv.LINE_AA);
    }
  }

  cv.imshow("output", dst);
  setTimeout(processVideo, 0);
}

main();