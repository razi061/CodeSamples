<?php

error_reporting(E_ERROR | E_PARSE);
ini_set('display_errors', '1');
ini_set("memory_limit", "2000M");
ini_set("max_execution_time", "150");

require 'vendor/autoload.php';

use Phpml\Dataset\CsvDataset;
use Phpml\Classification\MLPClassifier;
use Phpml\ModelManager;
use Phpml\NeuralNetwork\ActivationFunction\Sigmoid;
use Phpml\Metric\ClassificationReport;
use Phpml\Preprocessing\Normalizer;
use Phpml\Preprocessing\Imputer;
use Phpml\Preprocessing\Imputer\Strategy\MeanStrategy;
use Phpml\Metric\Regression;
use Phpml\Regression\LeastSquares;

$csvFiles = [
    '091107_Cleaned_Wynyard_9 km.csv', 
    '091237_Cleaned_Launceston_1.8 km.csv', 
    '091292_Cleaned_Smithton_3 km.csv', 
    '094029_Cleaned_Hobart_6 km.csv', 
    '094212_Cleaned_Campania_14 km.csv'
];

function categorize($value, $bins) {
    foreach ($bins as $index => $bin) {
        if ($value <= $bin) {
            return $index;
        }
    }
    return count($bins); 
}

$data = [];
$finalData = [];
$num = 0;

$year = 0;
$month = 0;
$day = 0;
$hour = 0;
$minute = 0;
$key = "";

foreach ($csvFiles as $file) {
    $dataset = new CsvDataset($file, 4, true);

    foreach ($dataset->getSamples() as $index => $sample) { 
        if(empty($sample[1])) {
            //$num = $num + 1;
            continue;
        }

        $year = date('Y', strtotime($sample[1]));
        $month = date('m', strtotime($sample[1]));
        $day = date('d', strtotime($sample[1]));
        $hour = date('H', strtotime($sample[1]));
        $minute = date('i', strtotime($sample[1]));
        $key = strval($sample[0]) . strval($year) . strval($month) . strval($day);

        if($minute == 0 || $minute == 30) {
            $d = [$key, $sample[0], $year, $month, $day, $hour, $minute, $sample[3], $sample[2]];
            array_push($data, $d);
            //$d = [$sample[2], $sample[3]];
            //array_push($labels, $d);
        }
    }

    unset($dataset);
}

echo "<h1>Data merge complete</h1>";

$imputer = new Imputer(null, new MeanStrategy(), Imputer::AXIS_COLUMN);
$imputer->fit($data);
$imputer->transform($data);
unset($imputer);

echo "<h1>Data imputation complete</h1>";

$summaryData = [];

foreach($data as $dr) {
    $key = $dr[0];

    if(!isset($summaryData[$key])) {
        $summaryData[$key] = [$dr[7], $dr[7], $dr[8], $dr[8]];
    } else {
        $summaryData[$key][0] = min($summaryData[$key][0], $dr[7]);
        $summaryData[$key][1] = max($summaryData[$key][1], $dr[7]);
        $summaryData[$key][2] = min($summaryData[$key][2], $dr[8]);
        $summaryData[$key][3] = max($summaryData[$key][3], $dr[8]);
    }
}

foreach($data as $dr) {
    $key = $dr[0];
    $d = [$dr[1], $dr[2], $dr[3], $dr[4], $dr[5], $dr[6], $dr[7], $dr[8], $summaryData[$key][0], $summaryData[$key][1], $summaryData[$key][2], $summaryData[$key][3]];
    array_push($finalData, $d);
}

unset($data);
unset($summaryData);

$data = [];
$lbl_MinTemp = [];
$lbl_MaxTemp = [];
$lbl_MinHum = [];
$lbl_MaxHum = [];

foreach($finalData as $dr) {
    $d = [(int)$dr[0], (int)$dr[1], (int)$dr[2], (int)$dr[3], (int)$dr[4], (int)$dr[5], (float)$dr[6], (float)$dr[7]];
    array_push($data, $d);

    array_push($lbl_MinTemp, (float)$dr[8]);
    array_push($lbl_MaxTemp, (float)$dr[9]);
    array_push($lbl_MinHum, (float)$dr[10]);
    array_push($lbl_MaxHum, (float)$dr[11]);
}

unset($finalData);

// $normalizer = new Normalizer();
// $normalizer->transform($data);
// unset($normalizer);

// echo "<h1>Data Normalization complete</h1>";

$splitRatio = 0.8;
$splitIndex = (int) (count($data) * $splitRatio);

$trainData = array_slice($data, 0, $splitIndex);
$testData = array_slice($data, $splitIndex);

$d = $testData[100];
$result = [];

$trainLabels = array_slice($lbl_MinTemp, 0, $splitIndex);
$testLabels = array_slice($lbl_MinTemp, $splitIndex);

unset($data);
unset($lbl_MinTemp);

$regression = new LeastSquares();
$regression->train($trainData, $trainLabels);
array_push($result, $regression->predict($d));
$modelManager = new ModelManager();
$modelManager->saveToFile($regression, 'wp_min_temp.model');
echo "<h3>Model trained for predicting minimum temperature and save to 'wp_min_temp.model'</h3>";

$predictions = array_map(function ($sample) use ($regression) {
    return $regression->predict($sample);
}, $testData);
unset($regression);

$mae = Regression::meanAbsoluteError($testLabels, $predictions);
$mse = Regression::meanSquaredError($testLabels, $predictions);
$r2 = Regression::r2Score($testLabels, $predictions);
unset($predictions);
unset($trainLabels);
unset($testLabels);

echo "Evaluation for Minimum Temperature Humidity prediction: <br />";
echo "MAE: $mae <br />";
echo "MSE: $mse <br />";
echo "R2: $r2 <br />";
echo "<hr />";

$trainLabels = array_slice($lbl_MaxTemp, 0, $splitIndex);
$testLabels = array_slice($lbl_MaxTemp, $splitIndex);

unset($lbl_MaxTemp);

$regression = new LeastSquares();
$regression->train($trainData, $trainLabels);
array_push($result, $regression->predict($d));
$modelManager = new ModelManager();
$modelManager->saveToFile($regression, 'wp_max_temp.model');
echo "<h3>Model trained for predicting maximum temperature and save to 'wp_max_temp.model'</h3>";

$predictions = array_map(function ($sample) use ($regression) {
    return $regression->predict($sample);
}, $testData);
unset($regression);

$mae = Regression::meanAbsoluteError($testLabels, $predictions);
$mse = Regression::meanSquaredError($testLabels, $predictions);
$r2 = Regression::r2Score($testLabels, $predictions);
unset($predictions);
unset($trainLabels);
unset($testLabels);

echo "Evaluation for Maximum Temperature prediction: <br />";
echo "MAE: $mae <br />";
echo "MSE: $mse <br />";
echo "R2: $r2 <br />";
echo "<hr />";

$trainLabels = array_slice($lbl_MinHum, 0, $splitIndex);
$testLabels = array_slice($lbl_MinHum, $splitIndex);

unset($lbl_MinHum);

$regression = new LeastSquares();
$regression->train($trainData, $trainLabels);
array_push($result, $regression->predict($d));
$modelManager = new ModelManager();
$modelManager->saveToFile($regression, 'wp_min_hum.model');
echo "<h3>Model trained for predicting minimum humidity and save to 'wp_min_hum.model'</h3>";

$predictions = array_map(function ($sample) use ($regression) {
    return $regression->predict($sample);
}, $testData);
unset($regression);

$mae = Regression::meanAbsoluteError($testLabels, $predictions);
$mse = Regression::meanSquaredError($testLabels, $predictions);
$r2 = Regression::r2Score($testLabels, $predictions);
unset($predictions);
unset($trainLabels);
unset($testLabels);

echo "Evaluation for Minimum Humidity prediction: <br />";
echo "MAE: $mae <br />";
echo "MSE: $mse <br />";
echo "R2: $r2 <br />";
echo "<hr />";

$trainLabels = array_slice($lbl_MaxHum, 0, $splitIndex);
$testLabels = array_slice($lbl_MaxHum, $splitIndex);

unset($lbl_MaxHum);

$regression = new LeastSquares();
$regression->train($trainData, $trainLabels);
array_push($result, $regression->predict($d));
$modelManager = new ModelManager();
$modelManager->saveToFile($regression, 'wp_max_hum.model');
echo "<h3>Model trained for predicting minimum humidity and save to 'wp_max_hum.model'</h3>";

$predictions = array_map(function ($sample) use ($regression) {
    return $regression->predict($sample);
}, $testData);
unset($regression);

$mae = Regression::meanAbsoluteError($testLabels, $predictions);
$mse = Regression::meanSquaredError($testLabels, $predictions);
$r2 = Regression::r2Score($testLabels, $predictions);
unset($predictions);
unset($trainLabels);
unset($testLabels);

echo "Evaluation for Maximum Humidity prediction: <br />";
echo "MAE: $mae <br />";
echo "MSE: $mse <br />";
echo "R2: $r2 <br />";
echo "<hr />";

echo "Sample prediction for: <br />";
echo "<pre>";
print_r($d);
echo "</pre>";
echo "<table><thead><tr><td>Minimum Temperature</td><td>Maximum Temperature</td><td>Minimum Humidity</td><td>Maximum Humidity</td></tr></thead>";
echo "<tr><td>" . $result[0] . "</td><td>" . $result[1] . "</td><td>" . $result[2] . "</td><td>" . $result[3] . "</td></tr></table>";
echo "<hr />";
echo "End of Task";

unset($trainData);
unset($testData);
