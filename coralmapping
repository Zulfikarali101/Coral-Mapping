// step 1 import dan memilih citra Sentinel 2A pada rentang waktu dan dan area beserta cloud mask
function maskS2clouds(image) {
  var qa = image.select('QA60');

  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000)
      .select("B.*")
      .copyProperties(image, ["system:time_start"]);
}

var sr2ACol = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate('2023-01-01', '2023-12-31')
    .filterBounds(area2)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskS2clouds)
    .median();
Map.centerObject(area1, 13.5);

// step 2 menampilkan citra pada peta
Map.addLayer(sr2ACol.clip(area1), imageVisParam, 'RGB');

// step 4 koreksi permukaan (Xij)
// step 4.1 rerata nilai laut dalam
var image = sr2ACol.select("B1", "B2", "B3", "B4", "B8").clip(area1);

var extent = ee.Feature(laut_dalam);
var deepAve = image.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: extent.geometry(),
  scale: 10,
  maxPixels: 1e9
});

print(deepAve);

// step 4.2 implementasi algoritma Xi = ln(laut dangkal - rerata laut dalam)
var X1cor = image.select(['B1']).subtract(0.145818).log().rename('X1');
var X2cor = image.select(['B2']).subtract(0.140087).log().rename('X2');
var X3cor = image.select(['B3']).subtract(0.128643).log().rename('X3');
var X4cor = image.select(['B4']).subtract(0.122570).log().rename('X4');

var Xij = X1cor.addBands(X2cor).addBands(X3cor).addBands(X4cor);

Map.addLayer(Xij,imageVisParam2, 'Image Koreksi Permukaan');

// step 5 implementasi algoritma Yij
// step 5.1 calculate ki/kj
var imgsand = Xij.clip(Sand);
var k23 = imgsand.select(['X2','X3'])
    .reduceRegion(ee.Reducer.linearRegression(1,1), imgsand.get('system:footprint'), 10);
print(k23);
var k24 = imgsand.select(['X2','X4'])
    .reduceRegion(ee.Reducer.linearRegression(1,1), imgsand.get('system:footprint'), 10);
print(k24);
var k34 = imgsand.select(['X3','X4'])
    .reduceRegion(ee.Reducer.linearRegression(1,1), imgsand.get('system:footprint'), 10);
print(k34);

// step 5.2 calculate depth invariance index
var Y23 = Xij.select(['X2']).subtract(Xij.select(['X3']).multiply(0.828411)).rename('Y23');
var Y24 = Xij.select(['X2']).subtract(Xij.select(['X4']).multiply(0.896653)).rename('Y24');
var Y34 = Xij.select(['X3']).subtract(Xij.select(['X4']).multiply(1.087374)).rename('Y34');

var Yij = Y23.addBands(Y24).addBands(Y34);

Map.addLayer(Yij, imageVisParam3, 'Indeks Kedalaman Yij');

// step 6 sampling data collection
// step 6.1 class "vegetasi", "pasir", "terumbu karang", "laut dalam"
var points = ee.FeatureCollection([
  ee.Feature(Sand, {class: 0}),
  ee.Feature(Coral, {class: 1}),
  ee.Feature(Lamun, {class: 2}),
  ee.Feature(Land, {class: 3}),
  ee.Feature(Laut, {class: 4})
  ]);
  
  // step 6.2 training and testingg data
var bands = ['B2', 'B3', 'B4', 'B8'];
var training = image.select(bands).sampleRegions({
  collection: points,
  properties: ['class'],
  scale: 10
});

var withRandom = training.randomColumn('random');
var split = 0.7; //70% training 30% testing
var trainingPoints = withRandom.filter(ee.Filter.lt('random', split));
var testingPoints = withRandom.filter(ee.Filter.lt('random', split));

print(testingPoints);

//Step 7 Klasifikasi
//SVM
var classifier = ee.Classifier.libsvm({
  kernelType: 'RBF',
  gamma: 0.5,
  cost:10
});
// Train the Classifier
var trained = classifier.train(trainingPoints, 'class', bands);

//classify image
var classified = image.classify(trained);
Map.addLayer(classified,
            {min:0, max: 4, palette: [ 'white', 'pink', 'yellow', 'green', 'blue']}, 
            'Terumbu Karang Supervised');
var validation = testingPoints.classify(trained);
print (validation);

//uji akurasi
var testAccuracy = validation.errorMatrix('class','classification');
print('Validation error matrix: ', testAccuracy);
print('Validation overall accuracy: ',testAccuracy.accuracy());

var OA = testAccuracy.accuracy();
var CA = testAccuracy.consumersAccuracy();
var Kappa = testAccuracy.kappa();
var Order = testAccuracy.order();
var PA = testAccuracy.producersAccuracy();

print(OA, 'Overall Accuracy');
print(CA, 'Consumers Accuracy');
print(Kappa, 'Kappa');
print(Order, 'Order');
print(PA, 'Producers Accuracy');

//Mengukur luasan
var class_areas = ee.Image.pixelArea().divide(1000*1000).addBands(classified).reduceRegion({
  reducer:ee.Reducer.sum().group({
    groupField:1,
    groupName:'code',
  }),
  geometry: area1,
  maxPixels: 500000000,
  scale: 10,
}).get('groups');

print(class_areas);

//klasifikasi unsupervised
var training = Yij.sample({
region: area1,
scale: 30,
numPixels: 5000
});
var clusterer = ee.Clusterer.wekaKMeans(30).train(training);
var result2010 = Yij.cluster(clusterer);
Map.addLayer(result2010.randomVisualizer(), {}, 'Terumbu Karang Unsupervised');

// Anggap Anda memiliki gambar klasifikasi (`classifiedImage`)
var clusterToExtract = [18,9,5]; // Ubah ini ke ID cluster yang diinginkan
var clusterBand = result2010.select('cluster');
var mergedClusterMask = clusterBand.eq(clusterToExtract[0]);
for (var i = 1; i < clusterToExtract.length; i++) {
  mergedClusterMask = mergedClusterMask.or(clusterBand.eq(clusterToExtract[i]));
}
Map.addLayer(mergedClusterMask,{min:0,max: 3, palette:['white', 'yellow', 'green']},'Karang_Saja');

var luasankarang = ee.Image.pixelArea().divide(1000*1000).addBands(mergedClusterMask).reduceRegion({
  reducer:ee.Reducer.sum().group({
    groupField:1,
    groupName:'code',
  }),
  geometry: geometry,
  maxPixels: 500000000,
  scale: 10,
}).get('groups');

print(luasankarang);

// Export the image, specifying scale and region.
Export.image.toDrive({
  image: classified,
  description: 'Pemetaan Terumbu Karang Supervised',
  scale: 30,
  region: area1,
  fileFormat: 'GeoTIFF'
});

// Export the image, specifying scale and region.
Export.image.toDrive({
  image: result2010,
  description: 'Pemetaan Terumbu Karang Unsupervised',
  scale: 30,
  region: area1,
  fileFormat: 'GeoTIFF'
});
