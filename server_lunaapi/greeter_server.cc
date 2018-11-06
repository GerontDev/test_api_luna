/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <memory>
#include <string>
#include <fsdk/FaceEngine.h>

#include <grpcpp/grpcpp.h>
#include "test_api.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using LunaSDK::Image;
using LunaSDK::ImageProccessingResult;
using LunaSDK::LunaSDKServer;

// Logic and data behind the server's behavior.
class GreeterServiceImpl final : public LunaSDKServer::Service {
  Status Proccesing(ServerContext* context, const Image* request, ImageProccessingResult* reply) override
  {
    //std::string prefix("Hello ");
	if(request->image_date.size() == 0)
		return Status::INVALID_ARGUMENT;

		//reply->set_message(prefix + request->name());
		std::cout<< "Proccesing : Image len = "<<request->image_date_size()<< " size = { H"<< request->height()<<" : W"<<request->width() <<"}" <<std::endl;

		// Facial feature detection confidence threshold.
		//const float confidenceThreshold = 0.25f;

		// Create FaceEngine root SDK object.
		fsdk::IFaceEnginePtr faceEngine = fsdk::acquire(fsdk::createFaceEngine("./data", "./data/faceengine.conf"));
		if (!faceEngine) {
			std::cerr << "Failed to create face engine instance." << std::endl;
			return Status::INTERNAL;
		}

		// Create MTCNN detector.
		fsdk::IDetectorPtr faceDetector = fsdk::acquire(faceEngine->createDetector(fsdk::ODT_MTCNN));
		if (!faceDetector) {
			std::cerr << "Failed to create face detector instance." << std::endl;
			return  Status::INTERNAL;
		}

		// Create warper.
		fsdk::IWarperPtr warper = fsdk::acquire(faceEngine->createWarper());
		if (!warper) {
			std::cerr << "Failed to create face warper instance." << std::endl;
			return  Status::INTERNAL;
		}

		// Create attribute estimator.
		fsdk::IAttributeEstimatorPtr attributeEstimator = fsdk::acquire(faceEngine->createAttributeEstimator());
		if (!attributeEstimator) {
			std::cerr << "Failed to create attribute estimator instance." << std::endl;
			return  Status::INTERNAL;
		}

		// Create quality estimator.
		fsdk::IQualityEstimatorPtr qualityEstimator = fsdk::acquire(faceEngine->createQualityEstimator());
		if (!qualityEstimator) {
			std::cerr << "Failed to create quality estimator instance." << std::endl;
			return  Status::INTERNAL;
		}

		// Create head pose estimator.
		fsdk::IHeadPoseEstimatorPtr headPoseEstimator = fsdk::acquire(faceEngine->createHeadPoseEstimator());
		if (!headPoseEstimator) {
			std::cerr << "Failed to create head pose estimator instance." << std::endl;
			return  Status::INTERNAL;
		}

		// Create overlap estimator.
		fsdk::IOverlapEstimatorPtr overlapEstimator = fsdk::acquire(faceEngine->createOverlapEstimator());
		if (!overlapEstimator) {
			std::cerr << "Failed to create overlap estimator instance." << std::endl;
			return  Status::INTERNAL;
		}

		// Load image.
		fsdk::Image image;
		if (!image.load(imagePath, fsdk::Format::R8G8B8)) {
			std::cerr << "Failed to load image: \"" << imagePath << "\"" << std::endl;
			return  Status::INTERNAL;
		}

		std::clog << "Detecting faces." << std::endl;

		// Detect no more than 10 faces in the image.
		enum { MaxDetections = 10 };

		// Data used for detection.
		fsdk::Detection detections[MaxDetections];
		int detectionsCount(MaxDetections);
		fsdk::Landmarks5 landmarks5[MaxDetections];
		fsdk::Landmarks68 landmarks68[MaxDetections];

		// Detect faces in the image.
		fsdk::ResultValue<fsdk::FSDKError, int> detectorResult = faceDetector->detect(
			image,
			image.getRect(),
			&detections[0],
			&landmarks5[0],
			&landmarks68[0],
			detectionsCount
		);

		if (detectorResult.isError()) {
			std::cerr << "Failed to detect face detection. Reason: " << detectorResult.what() << std::endl;
			return Status::INTERNAL;
		}

		detectionsCount = detectorResult.getValue();
		if (detectionsCount == 0) {
			std::clog << "Faces is not found." << std::endl;
			return Status::OK;
		}
		std::clog << "Found " << detectionsCount << " face(s)." << std::endl;

		// Loop through all the faces.
		for (int detectionIndex = 0; detectionIndex < detectionsCount; ++detectionIndex) {

			::LunaSDK::FaceFountAttribute* face_ = reply->add_facefounts();

			::LunaSDK::Rectangle* rect = new ::LunaSDK::Rectangle();
			rect->set_height(detections[detectionIndex].rect.height);
			rect->set_width(detections[detectionIndex].rect.width);
			rect->set_x(detections[detectionIndex].rect.x);
			rect->set_y(detections[detectionIndex].rect.y);
			face_->set_allocated_rect(rect);

			std::cout << "Detection " << detectionIndex + 1 <<
				"\nRect: x=" << detections[detectionIndex].rect.x <<
				" y=" << detections[detectionIndex].rect.y <<
				" w=" << detections[detectionIndex].rect.width <<
				" h=" << detections[detectionIndex].rect.height << std::endl;

			face_->set_score(detections[detectionIndex].score);

			//// Estimate confidence score of face detection.
			//if (detections[detectionIndex].score < confidenceThreshold) {
			//	std::clog << "Face detection succeeded, but confidence score of detection is small." << std::endl;
			//	continue;
			//}

			// Get warped face from detection.
			fsdk::Transformation transformation;
			fsdk::Landmarks5 transformedLandmarks5;
			fsdk::Landmarks68 transformedLandmarks68;
			fsdk::Image warp;
			transformation = warper->createTransformation(detections[detectionIndex], landmarks5[detectionIndex]);
			fsdk::Result<fsdk::FSDKError> transformedLandmarks5Result = warper->warp(
				landmarks5[detectionIndex],
				transformation,
				transformedLandmarks5
			);
			if (transformedLandmarks5Result.isError()) {
				std::cerr << "Failed to create transformed landmarks5. Reason: " <<
					transformedLandmarks5Result.what() << std::endl;
				return -1;
			}
			fsdk::Result<fsdk::FSDKError> transformedLandmarks68Result = warper->warp(
				landmarks68[detectionIndex],
				transformation,
				transformedLandmarks68
			);
			if (transformedLandmarks68Result.isError()) {
				std::cerr << "Failed to create transformed landmarks68. Reason: " <<
					transformedLandmarks68Result.what() << std::endl;
				return -1;
			}
			fsdk::Result<fsdk::FSDKError> warperResult = warper->warp(image, transformation, warp);
			if (warperResult.isError()) {
				std::cerr << "Failed to create warped face. Reason: " << warperResult.what() << std::endl;
				return -1;
			}

			// Save warped face.
			warp.save(("warp_" + std::to_string(detectionIndex) + ".ppm").c_str());
			LunaSDK::Image warpImage = new  LunaSDK::Image();

			face_->set_allocated_warpiamge(warpImage);

			// Get attribute estimate.
			fsdk::AttributeEstimation attributeEstimation;
			fsdk::Result<fsdk::FSDKError> attributeEstimatorResult = attributeEstimator->estimate(warp, attributeEstimation);
			if (attributeEstimatorResult.isError()) {
				std::cerr << "Failed to create attribute estimation. Reason: " << attributeEstimatorResult.what() << std::endl;
				return Status::INTERNAL;
			}
			std::cout << "\nAttribure estimate:" <<
				"\ngender: " << attributeEstimation.gender << " (1 - man, 0 - woman)"
				"\nglasses: " << attributeEstimation.glasses <<
				" (1 - person wears glasses, 0 - person doesn't wear glasses)" <<
				"\nage: " << attributeEstimation.age << " (in years)" << std::endl;

			// Get quality estimate.
			fsdk::Quality qualityEstimation;
			fsdk::Result<fsdk::FSDKError> qualityEstimationResult = qualityEstimator->estimate(warp, qualityEstimation);
			if (qualityEstimationResult.isError()) {
				std::cerr << "Failed to create quality estimation. Reason: " << qualityEstimationResult.what() << std::endl;
				return Status::INTERNAL;
			}

			LunaSDK::QualityFaceFountAttribute * Quality = new LunaSDK::QualityFaceFountAttribute();
			Quality->set_ligth(qualityEstimation.light);
			Quality->set_dark(qualityEstimation.dark);
			Quality->set_gray(qualityEstimation.gray);
			Quality->set_blur(qualityEstimation.blur);
			Quality->set_quality(qualityEstimation.getQuality());

			std::cout << "Quality estimate:" <<
				"\nlight: " << qualityEstimation.light <<
				"\ndark: " << qualityEstimation.dark <<
				"\ngray: " << qualityEstimation.gray <<
				"\nblur: " << qualityEstimation.blur <<
				"\nquality: " << qualityEstimation.getQuality() << std::endl;

			// Get head pose estimate.
			fsdk::HeadPoseEstimation headPoseEstimation;
			fsdk::Result<fsdk::FSDKError> headPoseEstimationResult = headPoseEstimator->estimate(
				image,
				detections[detectionIndex],
				headPoseEstimation
			);
			if (headPoseEstimationResult.isError()) {
				std::cerr << "Failed to create head pose estimation. Reason: " << headPoseEstimationResult.what() << std::endl;
				return Status::INTERNAL;
			}
			LunaSDK::HeadPoseFaceFountAttribute* HeadPose = new LunaSDK::HeadPoseFaceFountAttribute();
			HeadPose->set_pitch(headPoseEstimation.pitch);
			HeadPose->set_rool(headPoseEstimation.yaw);
			HeadPose->set_yam(headPoseEstimation.roll);

			std::cout << "Head pose estimate:" <<
				"\npitch angle estimation: " << headPoseEstimation.pitch <<
				"\nyaw angle estimation: " << headPoseEstimation.yaw <<
				"\nroll angle estimation: " << headPoseEstimation.roll <<
				std::endl;
			std::cout << std::endl;

			// Get overlap estimation.
			fsdk::OverlapEstimation overlapEstimation;
			fsdk::Result<fsdk::FSDKError> overlapEstimationResult = overlapEstimator->estimate(
				image, detections[0], overlapEstimation);

			if (overlapEstimationResult.isError()) {
				std::cerr << "Failed overlap estimation. Reason: " << overlapEstimationResult.what() << std::endl;
				return Status::INTERNAL;
			}

			std::cout << "Face overlap estimate:"
				<< "\noverlapValue: " << overlapEstimation.overlapValue << " (range [0, 1])"
				<< "\noverlapped: " << overlapEstimation.overlapped << " (0 - not overlapped, 1 - overlapped)"
				<< std::endl;
		}

		return Status::OK;   
  }
};

void RunServer() {
  std::string server_address("0.0.0.0:50051");
  GreeterServiceImpl service;

  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {
  RunServer();

  return 0;
}
