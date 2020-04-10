# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Assignment Report
<img src="doc/result.png" width="1232" height="369" />

### Rubrics
#### FP.1 Match 3D Objects

 <span style="color:blue">
"Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences."
</span>

```cpp
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // use the matched keypoints to identify the matching between bounding boxes-
    // first, associate the keypoints to detected bounding boxes and then check which bounding boxes are relevant.
    
    // loop through all keypoint matches and assign them the bounding boxes they belong to.
    
    cv::Mat bbMatchTable = cv::Mat::zeros(prevFrame.boundingBoxes.size(), currFrame.boundingBoxes.size(), CV_32S);

    for (const auto &match : matches)
    {
        const cv::KeyPoint &currKpt = currFrame.keypoints[ match.trainIdx ];
        const cv::KeyPoint &prevKpt = prevFrame.keypoints[ match.queryIdx ];

        for (const BoundingBox& prevBoundingBox : prevFrame.boundingBoxes)
        {
            for (const BoundingBox& currBoundingBox : currFrame.boundingBoxes)
            {
                if (prevBoundingBox.roi.contains(prevKpt.pt) && currBoundingBox.roi.contains(currKpt.pt))
                {
                    bbMatchTable.at<int>(prevBoundingBox.boxID, currBoundingBox.boxID)++;
                }
            }
        }
    }  

    // loop through the prev frame counts
    for (int i = 0; i < bbMatchTable.rows; i++)
    {
        int bestMatchCounts = 0;
        int bestMatchIndex = -1;

        // loop through the curr frame counts
        for (int j = 0; j < bbMatchTable.cols; j++)
        {
            if (bbMatchTable.at<int>(i, j) > 0 && bbMatchTable.at<int>(i, j) > bestMatchCounts)
            {
                bestMatchCounts = bbMatchTable.at<int>(i, j);
                bestMatchIndex = j;
            }
        }

        if (bestMatchIndex != -1)
        {
            bbBestMatches.emplace(i, bestMatchIndex);
        }
    }
}
```

#### FP.2 Compute Lidar-based TTC

 <span style="color:blue">
"Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame."
</span>

```cpp
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // find closest distance to Lidar points 
    double prevXMean = 0.0;
    double currXMean = 0.0;

    // calculate the mean of the lidar points x distance to minimize wrong TTC estimation
    // error when outliers are involved in the calculation.

    for(auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        prevXMean += it->x;
    }
    prevXMean /= lidarPointsPrev.size();

    for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        currXMean += it->x;
    }
    currXMean /= lidarPointsCurr.size();

    // compute TTC from both measurements
    const double sampleTime_s = 1.0 / frameRate;
    TTC = currXMean * sampleTime_s / (prevXMean - currXMean);
}
```

#### FP.3 Associate Keypoint Correspondences with Bounding Boxes

 <span style="color:blue">
"Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box."
</span>

```cpp
// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    float meanDistance = 0.0;
    int numValidPts = 0;

    for (auto kptMatchItr = kptMatches.begin(); kptMatchItr != kptMatches.end(); kptMatchItr++)
    {
        cv::KeyPoint keypoint = kptsCurr[kptMatchItr->trainIdx];

        if (boundingBox.roi.contains(keypoint.pt))
        {
            meanDistance += kptMatchItr->distance;
            numValidPts++;
        }
    }
    meanDistance /= numValidPts;

    for (auto kptMatchItr = kptMatches.begin(); kptMatchItr != kptMatches.end(); kptMatchItr++)
    {
        cv::KeyPoint keypoint = kptsCurr[kptMatchItr->trainIdx];

        if (boundingBox.roi.contains(keypoint.pt) && (kptMatchItr->distance < meanDistance))
        {
            boundingBox.kptMatches.push_back(*kptMatchItr);
            boundingBox.keypoints.push_back(keypoint);
        }
    }
}
```

#### FP.4 Compute Camera-based TTC

 <span style="color:blue">
"Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame."
</span>

```cpp
// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    const bool isEven = (distRatios.size() % 2 == 0) ? true : false;
    const int medianIndex = floor((distRatios.size() / 2));

    double medianDistRatio;
    if (isEven)
    {
        medianDistRatio = distRatios[medianIndex-1] + distRatios[medianIndex];
        medianDistRatio /= 2.0;
    }
    else
    {
        medianDistRatio = distRatios[medianIndex];
    }
    
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);
}
```

#### FP.5 Performance Evaluation 1

 <span style="color:blue">
"Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened."</span>

When the detected Lidar points are distributed on wide on x-direction, the method of averaging the detected points will cause the TTC to jump as in the situation captured below. This could happen because of the calculated mean shifted far away than the actual closest point which will estimate more time for collision as the preceding vehicle get more far away.

<img src="doc/lidar_detection.png" width="1232" height="369" />

The reason could be because of the Lidar high resolution which makes it able to capture the curved edges of the vehicle which we are not interested in for this application. 

#### FP.6 Performance Evaluation 2

<span style="color:blue">
"Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons."</span>

I have checked with several combinations and presented below a table with the values. I was mainly focusing of detecting the frame at which max difference between the TTC calculated from lidar and camera occured. So that we can see what's the maximum difference compared with several detector and descriptor combinations. And found that the FAST/BRIEF and SIFT/SIFT have the least difference in TTC computed. Besides I presented also the average of the TTC calculated along the whole set of frames to ensure there is no much deviation to the lidar which might indicate the algorithm is not functioning correctly.

|Detector| Descriptor | Prev Image Frame  | Current Image Frame | Lidar TTC | Camera TTC | Difference in TTC | Lidar TTC Average | Camera TTC Average |
|--------|------------|------------------|---------------------|-----------|------------|-------------------|-------------------|-------------------|
|SHITOMASI |BRISK|12|13|9.42504|13.3278|3.90278|11.7444|12.1111|
|FAST|BRIEF|17|18|8.53557|12.1254|3.58979|11.7444|12.1451|
|FAST|ORB|11|12|9.88711|15.1852|5.29804|11.7444|12.5039|
|AKAZE|AKAZE|2|3|16.3845|12.6936|3.69089|11.7444|12.2384|
|SIFT|SIFT|2|3|16.3845|13.0962|3.28828|11.7444|11.6279|
|FAST|FREAK|4|5|12.7299|5.98282|6.74713|11.7444|11.8134|
