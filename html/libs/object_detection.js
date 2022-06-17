var player = document.getElementById('player')
var drawCanvas = document.createElement('canvas');
player.appendChild(drawCanvas)
// document.body.appendChild(drawCanvas);
drawCanvas.setAttribute('id', 'drawCanvas')
var drawCtx = drawCanvas.getContext("2d");
var imageCanvas = document.getElementsByTagName('canvas')[0];
var imageWidth = imageCanvas.width,
    imageHeight = imageCanvas.height;

var getPixelRatio = function (context) {
    var backingStore = context.backingStorePixelRatio ||
        context.webkitBackingStorePixelRatio ||
        context.mozBackingStorePixelRatio ||
        context.msBackingStorePixelRatio ||
        context.oBackingStorePixelRatio ||
        context.backingStorePixelRatio || 1;
    return (window.devicePixelRatio || 1) / backingStore;
};
var ratio = getPixelRatio(drawCanvas);
drawCanvas.width = drawCanvas.width * ratio
drawCanvas.height = drawCanvas.height * ratio
drawCtx.setTransform(ratio, 0, 0, ratio, 0, 0)



//draw boxes and labels on each detected object
function drawBoxes(objects) {
    // clear the previous drawings
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);

    // filter out objects that contain a class_name and then draw boxes and labels on each
    objects.filter(object => object.class_name).forEach(object => {

        let x = object.x * imageWidth;
        let y = object.y * imageHeight;
        let width = (object.width * imageWidth) - x;
        let height = (object.height * imageHeight) - y;

        // flip the x axis if local video is mirrored
        // if (mirror) {
        //     x = imageWidth - (x + width)
        // }
        // console.log(x, y, width, height)

        drawCtx.lineWidth = 1;
        drawCtx.strokeStyle = "cyan";
        drawCtx.font = "7px Verdana";
        drawCtx.fillStyle = "cyan";


        drawCtx.fillText(object.class_name, x + 5, y + 10);
        drawCtx.strokeRect(x, y, width, height);

    });
}
