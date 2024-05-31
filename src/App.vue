<template>
  <main
    class="w-full min-h-screen main-bg text-white flex flex-col items-center justify-center gap-8 py-12"
  >
    <img src="./assets/logo.png" class="w-40" />
    <div class="flex flex-col items-center gap-3 text-center">
      <h1 class="text-4xl font-bold">Fashion Image Classifier</h1>
      <h2 class="text-3xl font-bold">Machine learning class</h2>
      <h3>Mateo Echeverri - Alejandro Realpe</h3>
    </div>
    <div class="flex flex-col items-center gap-3">
      <div class="w-full flex items-center justify-center gap-4">
        <p
          v-if="resultDNN"
          class="px-12 py-3 bg-white text-black rounded-md text-sm font-bold text-center"
        >
          Result DNN Cassification: <br />
          <span class="text-2xl text-blue-950">{{ resultDNN }}</span>
        </p>
        <p
          v-if="resultCNN"
          class="px-12 py-3 bg-white text-black rounded-md text-sm font-bold text-center"
        >
          Result CNN Cassification: <br />
          <span class="text-2xl text-blue-950">{{ resultCNN }}</span>
        </p>
      </div>
      <p>Draw here:</p>
    </div>
    <canvas class="bg-white" ref="bigCanvas" width="300" height="300"></canvas>
    <canvas class="hidden" ref="smallCanvas" width="28" height="28"></canvas>
    <div class="w-full flex items-center justify-center gap-5">
      <button
        @click="clean"
        class="px-8 py-3 rounded-lg bg-blue-950 hover:scale-95 transition-transform"
      >
        Clear
      </button>
      <button
        @click="classify"
        class="px-8 py-3 rounded-lg bg-blue-950 hover:scale-95 transition-transform"
      >
        Classify
      </button>
    </div>
  </main>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import * as tf from "@tensorflow/tfjs";

const resultDNN = ref<string | null>(null);
const resultCNN = ref<string | null>(null);
const bigCanvas = ref<HTMLCanvasElement | null>(null);
const smallCanvas = ref<HTMLCanvasElement | null>(null);
const bigCanvasCtx = ref<CanvasRenderingContext2D | null>(null);
const smallCanvasCtx = ref<CanvasRenderingContext2D | null>(null);

let DNNmodel: tf.LayersModel;
let CNNmodel: tf.LayersModel;
let initialX = 0;
let initialY = 0;

const CLASES = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
];

onMounted(async () => {
  bigCanvasCtx.value = bigCanvas.value?.getContext("2d") ?? null;
  smallCanvasCtx.value = smallCanvas.value?.getContext("2d") ?? null;

  bigCanvas.value?.addEventListener("mousedown", mouseDown);
  bigCanvas.value?.addEventListener("mouseup", mouseUp);

  DNNmodel = await tf.loadLayersModel("/dnn/model.json");
  CNNmodel = await tf.loadLayersModel("/cnn/model.json");
});

onUnmounted(() => {
  bigCanvas.value?.removeEventListener("mousedown", mouseDown);
  bigCanvas.value?.removeEventListener("mouseup", mouseUp);
});

const draw = (cursorX: number, cursorY: number) => {
  if (!bigCanvasCtx.value) return;

  bigCanvasCtx.value.beginPath();
  bigCanvasCtx.value.moveTo(initialX, initialY);
  bigCanvasCtx.value.lineWidth = 20;
  bigCanvasCtx.value.strokeStyle = "#000";
  bigCanvasCtx.value.lineCap = "round";
  bigCanvasCtx.value.lineJoin = "round";
  bigCanvasCtx.value.lineTo(cursorX, cursorY);
  bigCanvasCtx.value.stroke();

  initialX = cursorX;
  initialY = cursorY;
};

const mouseDown = (e: MouseEvent) => {
  initialX = e.offsetX;
  initialY = e.offsetY;
  draw(initialX, initialY);
  bigCanvas.value?.addEventListener("mousemove", mouseMoving);
};

const mouseMoving = (e: MouseEvent) => {
  draw(e.offsetX, e.offsetY);
};

const mouseUp = () => {
  bigCanvas.value?.removeEventListener("mousemove", mouseMoving);
};

const clean = () => {
  bigCanvasCtx.value?.clearRect(
    0,
    0,
    bigCanvas.value?.width ?? 0,
    bigCanvas.value?.height ?? 0
  );

  resultDNN.value = null;
  resultCNN.value = null;
};

const classify = () => {
  if (!bigCanvas.value || !smallCanvas.value) return;

  resampleSingle(bigCanvas.value, 28, 28, smallCanvas.value);

  var imgData = smallCanvasCtx.value!.getImageData(0, 0, 28, 28);
  let arr = []; //El arreglo completo
  let arr28 = []; //Al llegar a 28 posiciones se pone en 'arr' como un nuevo indice
  for (let i = 0; i < imgData.data.length; i += 4) {
    var valor = imgData.data[i + 3] / 255;
    arr28.push([valor]); //Agregar al arr28 y normalizar a 0-1. Aparte queda dentro de un arreglo en el indice 0... again
    if (arr28.length == 28) {
      arr.push(arr28);
      arr28 = [];
    }
  }

  arr = [arr]; //Meter el arreglo en otro arreglo por que si no tio tensorflow se enoja >:(
  //Nah basicamente Debe estar en un arreglo nuevo en el indice 0, por ser un tensor4d en forma 1, 28, 28, 1
  let tensor4 = tf.tensor4d(arr);
  try {
    // @ts-ignore
    const resultsDNN = DNNmodel.predict(tensor4).dataSync();
    const maxIndexDNN = resultsDNN.indexOf(Math.max.apply(null, resultsDNN));
    resultDNN.value = CLASES[maxIndexDNN];

    // @ts-ignore
    const resultsCNN = CNNmodel.predict(tensor4).dataSync();
    const maxIndexCNN = resultsCNN.indexOf(Math.max.apply(null, resultsCNN));
    resultCNN.value = CLASES[maxIndexCNN];
  } catch (error) {
    console.log({ error });
  }
};

const resampleSingle = (
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  resize_canvas: HTMLCanvasElement
) => {
  var width_source = canvas.width;
  var height_source = canvas.height;
  width = Math.round(width);
  height = Math.round(height);

  var ratio_w = width_source / width;
  var ratio_h = height_source / height;
  var ratio_w_half = Math.ceil(ratio_w / 2);
  var ratio_h_half = Math.ceil(ratio_h / 2);

  var ctx = canvas.getContext("2d");
  var ctx2 = resize_canvas.getContext("2d");
  var img = ctx!.getImageData(0, 0, width_source, height_source);
  var img2 = ctx2!.createImageData(width, height);
  var data = img.data;
  var data2 = img2.data;

  for (var j = 0; j < height; j++) {
    for (var i = 0; i < width; i++) {
      var x2 = (i + j * width) * 4;
      var weight = 0;
      var weights = 0;
      var weights_alpha = 0;
      var gx_r = 0;
      var gx_g = 0;
      var gx_b = 0;
      var gx_a = 0;
      var center_y = (j + 0.5) * ratio_h;
      var yy_start = Math.floor(j * ratio_h);
      var yy_stop = Math.ceil((j + 1) * ratio_h);
      for (var yy = yy_start; yy < yy_stop; yy++) {
        var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
        var center_x = (i + 0.5) * ratio_w;
        var w0 = dy * dy; //pre-calc part of w
        var xx_start = Math.floor(i * ratio_w);
        var xx_stop = Math.ceil((i + 1) * ratio_w);
        for (var xx = xx_start; xx < xx_stop; xx++) {
          var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
          var w = Math.sqrt(w0 + dx * dx);
          if (w >= 1) {
            //pixel too far
            continue;
          }
          //hermite filter
          weight = 2 * w * w * w - 3 * w * w + 1;
          var pos_x = 4 * (xx + yy * width_source);
          //alpha
          gx_a += weight * data[pos_x + 3];
          weights_alpha += weight;
          //colors
          if (data[pos_x + 3] < 255) weight = (weight * data[pos_x + 3]) / 250;
          gx_r += weight * data[pos_x];
          gx_g += weight * data[pos_x + 1];
          gx_b += weight * data[pos_x + 2];
          weights += weight;
        }
      }
      data2[x2] = gx_r / weights;
      data2[x2 + 1] = gx_g / weights;
      data2[x2 + 2] = gx_b / weights;
      data2[x2 + 3] = gx_a / weights_alpha;
    }
  }

  //Ya que esta, exagerarlo. Blancos blancos y negros negros..?

  for (var p = 0; p < data2.length; p += 4) {
    var gris = data2[p]; //Esta en blanco y negro

    if (gris < 100) {
      gris = 0; //exagerarlo
    } else {
      gris = 255; //al infinito
    }

    data2[p] = gris;
    data2[p + 1] = gris;
    data2[p + 2] = gris;
  }

  ctx2!.putImageData(img2, 0, 0);
};
</script>

<style>
.main-bg {
  background: rgb(2, 0, 36);
  background: linear-gradient(
    180deg,
    rgba(2, 0, 36, 1) 0%,
    rgba(9, 9, 121, 1) 70%,
    rgba(0, 212, 255, 1) 100%
  );
}
</style>
