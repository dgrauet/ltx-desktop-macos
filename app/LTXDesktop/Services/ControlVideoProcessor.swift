import Foundation
import AVFoundation
import CoreImage
import CoreVideo
import VideoToolbox

enum ControlType: String, CaseIterable, Identifiable {
    case raw, canny, pose, depth
    var id: String { rawValue }

    var label: String {
        switch self {
        case .raw: return "Raw (as-is)"
        case .canny: return "Canny edges"
        case .pose: return "Pose"
        case .depth: return "Depth"
        }
    }

    /// Pose/Depth are extracted on-device in Swift before submission.
    /// Raw submits the source video unchanged; Canny is extracted by the backend.
    var needsSwiftPreprocess: Bool { self == .pose || self == .depth }
}

enum ControlProcessingError: Error { case unreadable, writeFailed, unsupportedType }

/// Converts a normal video into an IC-LoRA control video (pose skeleton / depth map),
/// written to a temp mp4 at source dimensions and fps. The library resizes the control
/// video to the reference size itself, so we do not pre-downscale.
actor ControlVideoProcessor {

    private let ciContext = CIContext()
    private var depthExtractor: DepthExtractor?

    func process(_ inputURL: URL, type: ControlType,
                 progress: @escaping (Double) -> Void) async throws -> URL {
        let asset = AVURLAsset(url: inputURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ControlProcessingError.unreadable
        }
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let size = naturalSize.applying(transform)
        let width = Int(abs(size.width)), height = Int(abs(size.height))
        let fps = try await track.load(.nominalFrameRate)
        let totalFrames = try await estimateFrameCount(asset: asset, track: track, fps: fps)

        if type == .depth {
            let model = try await CoreMLModelManager.shared.depthModel(progress: progress)
            depthExtractor = try DepthExtractor(model: model)
        }
        defer { depthExtractor = nil }

        // Reader
        let reader = try AVAssetReader(asset: asset)
        let readerOutput = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String:
                                kCVPixelFormatType_32BGRA])
        reader.add(readerOutput)

        // Writer
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("control_\(UUID().uuidString).mp4")
        let writer = try AVAssetWriter(outputURL: outURL, fileType: .mp4)
        let bitrate = max(12_000_000, width * height * 4)  // high bitrate for thin lines
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: bitrate],
        ])
        writerInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ])
        writer.add(writerInput)

        guard reader.startReading(), writer.startWriting() else {
            throw ControlProcessingError.writeFailed
        }
        writer.startSession(atSourceTime: .zero)

        var frameIndex = 0
        while let sample = readerOutput.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let srcBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            let cg = try transformFrame(srcBuffer, type: type)
            let outBuffer = try makePixelBuffer(from: cg, width: width, height: height)
            while !writerInput.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 5_000_000)
            }
            adaptor.append(outBuffer, withPresentationTime: pts)
            frameIndex += 1
            // totalFrames is an estimate (duration × fps); the real decoded count
            // can exceed it, so clamp to avoid an out-of-range progress value.
            if totalFrames > 0 { progress(min(1.0, Double(frameIndex) / Double(totalFrames))) }
        }

        writerInput.markAsFinished()
        await writer.finishWriting()
        if writer.status != .completed { throw ControlProcessingError.writeFailed }
        return outURL
    }

    func transformFrame(_ pixelBuffer: CVPixelBuffer, type: ControlType) throws -> CGImage {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        switch type {
        case .pose:
            return try PoseRenderer.skeleton(from: pixelBuffer, width: width, height: height)
        case .depth:
            guard let extractor = depthExtractor else { throw ControlProcessingError.unsupportedType }
            return try extractor.depthMap(from: pixelBuffer, width: width, height: height)
        default:
            let ci = CIImage(cvPixelBuffer: pixelBuffer)
            guard let cg = ciContext.createCGImage(ci, from: ci.extent) else {
                throw ControlProcessingError.writeFailed
            }
            return cg
        }
    }

    private func makePixelBuffer(from cg: CGImage, width: Int, height: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let buffer = pb else { throw ControlProcessingError.writeFailed }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue)
        ctx?.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    private func estimateFrameCount(asset: AVAsset, track: AVAssetTrack,
                                    fps: Float) async throws -> Int {
        let dur = try await asset.load(.duration)
        return max(0, Int(CMTimeGetSeconds(dur) * Double(fps)))
    }
}
