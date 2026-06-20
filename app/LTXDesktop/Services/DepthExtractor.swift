import Foundation
import Vision
import CoreML
import CoreGraphics
import CoreVideo

/// Runs the Depth-Anything Core ML model on a frame and renders a grayscale depth map.
/// Vision resizes the input to the model's fixed size; the output is normalized per
/// frame (min–max) and drawn at the source dimensions.
final class DepthExtractor {
    private let vnModel: VNCoreMLModel

    init(model: MLModel) throws {
        self.vnModel = try VNCoreMLModel(for: model)
    }

    func depthMap(from pixelBuffer: CVPixelBuffer,
                  width: Int, height: Int) throws -> CGImage {
        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill  // stretch to model input
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])

        // Depth Anything outputs a single-channel depth map. Accept either a
        // pixel-buffer observation or a MultiArray observation.
        if let pbObs = request.results?.first as? VNPixelBufferObservation {
            return try normalizeToGray(pbObs.pixelBuffer, width: width, height: height)
        }
        guard let arrObs = request.results?.first as? VNCoreMLFeatureValueObservation,
              let arr = arrObs.featureValue.multiArrayValue else {
            throw ControlProcessingError.writeFailed
        }
        return try grayFromMultiArray(arr, width: width, height: height)
    }

    private func normalizeToGray(_ depth: CVPixelBuffer,
                                 width: Int, height: Int) throws -> CGImage {
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let dw = CVPixelBufferGetWidth(depth), dh = CVPixelBufferGetHeight(depth)
        let rowBytes = CVPixelBufferGetBytesPerRow(depth)
        guard let base = CVPixelBufferGetBaseAddress(depth) else {
            throw ControlProcessingError.writeFailed
        }
        let isFloat = CVPixelBufferGetPixelFormatType(depth)
            == kCVPixelFormatType_DepthFloat32 || CVPixelBufferGetPixelFormatType(depth)
            == kCVPixelFormatType_OneComponent32Float
        var values = [Float](repeating: 0, count: dw * dh)
        for y in 0..<dh {
            for x in 0..<dw {
                let v: Float
                if isFloat {
                    v = base.load(fromByteOffset: y * rowBytes + x * 4, as: Float.self)
                } else {
                    v = Float(base.load(fromByteOffset: y * rowBytes + x, as: UInt8.self))
                }
                values[y * dw + x] = v
            }
        }
        return try gray(values, srcW: dw, srcH: dh, outW: width, outH: height)
    }

    private func grayFromMultiArray(_ arr: MLMultiArray,
                                    width: Int, height: Int) throws -> CGImage {
        // Expect shape [..., H, W]; take the last two dims.
        let shape = arr.shape.map { $0.intValue }
        guard shape.count >= 2 else { throw ControlProcessingError.writeFailed }
        let h = shape[shape.count - 2], w = shape[shape.count - 1]
        var values = [Float](repeating: 0, count: w * h)
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: arr.count)
        let offset = arr.count - w * h
        for i in 0..<(w * h) { values[i] = ptr[offset + i] }
        return try gray(values, srcW: w, srcH: h, outW: width, outH: height)
    }

    private func gray(_ values: [Float], srcW: Int, srcH: Int,
                      outW: Int, outH: Int) throws -> CGImage {
        let lo = values.min() ?? 0, hi = values.max() ?? 1
        let range = max(hi - lo, 1e-6)
        var pixels = [UInt8](repeating: 0, count: srcW * srcH)
        for i in 0..<values.count {
            pixels[i] = UInt8(max(0, min(255, ((values[i] - lo) / range) * 255)))
        }
        let cs = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let small = CGImage(width: srcW, height: srcH, bitsPerComponent: 8,
                                  bitsPerPixel: 8, bytesPerRow: srcW, space: cs,
                                  bitmapInfo: CGBitmapInfo(rawValue: 0),
                                  provider: provider, decode: nil,
                                  shouldInterpolate: true, intent: .defaultIntent)
        else { throw ControlProcessingError.writeFailed }
        // Resize to output dims.
        guard let ctx = CGContext(data: nil, width: outW, height: outH, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw ControlProcessingError.writeFailed }
        ctx.interpolationQuality = .high
        ctx.draw(small, in: CGRect(x: 0, y: 0, width: outW, height: outH))
        guard let out = ctx.makeImage() else { throw ControlProcessingError.writeFailed }
        return out
    }
}
