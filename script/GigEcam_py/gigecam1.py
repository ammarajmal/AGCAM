#coding=utf-8
import cv2
import numpy as np
import mvsdk, time
import platform

def main_loop():
	# enumerate cameras
	DevList = mvsdk.CameraEnumerateDevice()
	nDev = len(DevList)
	if nDev < 1:
		print("No camera was found!")
		return

	for i, DevInfo in enumerate(DevList):
		print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
	i = 0 if nDev == 1 else int(input("Select camera: "))
	DevInfo = DevList[i]
	print(DevInfo)

	# Turn on the camera
	hCamera = 0
	try:
		hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
	except mvsdk.CameraException as e:
		print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
		return

	# Get camera feature description
	cap = mvsdk.CameraGetCapability(hCamera)

	# Determine whether it is a black and white camera or a color camera
	monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

	# Monochrome camera allows ISP to directly output MONO data 
 	# instead of expanding to 24-bit grayscale with R=G=B
	if monoCamera:
		mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
	else:
		mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

	# Camera mode switched to continuous acquisition
	mvsdk.CameraSetTriggerMode(hCamera, 0)

	# Manual exposure, exposure time 30ms
	mvsdk.CameraSetAeState(hCamera, 0)
	mvsdk.CameraSetExposureTime(hCamera, 8 * 1000)

	# Let the SDK internal image fetching thread start to work
	mvsdk.CameraPlay(hCamera)

	# Calculate the required size of the RGB buffer, 
 	# which is directly allocated according to the maximum resolution of the camera
	FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

	# Allocate RGB buffer to store the image output by ISP
	
 	# Remarks: RAW data is transmitted from the camera to the PC, 
 	# which is converted to RGB data by the software ISP on the PC 
	# (if it is a black and white camera, there is no need to convert the format,
	# but the ISP has other processing, so this buffer also needs to be allocated)
	pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
	pTime = time.time()
	while (cv2.waitKey(1) & 0xFF) != ord('q'):
		# Take a frame from the camera
		try:
			pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
			mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
			mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

			# Windows: The image data fetched under windows is upside down and stored in BMP format.
   			# To convert to opencv, you need to flip up and down to be positive
			
   			# Linux: Directly output positive under linux, no need to flip up and down
			if platform.system() == "Windows":
				mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
			
			# At this time, the picture has been stored in pFrameBuffer,
   			# for color camera pFrameBuffer=RGB data, for black and white camera pFrameBuffer=8-bit grayscale data
			# Convert pFrameBuffer to opencv image format for subsequent algorithm processing
			frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
			frame = np.frombuffer(frame_data, dtype=np.uint8)
			frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )

			frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LINEAR)
			cTime = time.time()
			fps = 1/(cTime-pTime)
			pTime = cTime
			cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
			print(int(fps))
			cv2.imshow("Press q to end", frame)
			
		except mvsdk.CameraException as e:
			if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
				print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )

	# turn off the camera
	mvsdk.CameraUnInit(hCamera)

	# free framebuffer
	mvsdk.CameraAlignFree(pFrameBuffer)

def main():
	try:
		main_loop()
	finally:
		cv2.destroyAllWindows()

main()
