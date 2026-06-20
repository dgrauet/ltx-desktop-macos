import Foundation
import Vision
import CoreGraphics
import CoreVideo

/// Renders an OpenPose-style skeleton (best-effort) from Vision body-pose joints.
/// Colors/connections live in one table so they can be tuned without refactoring.
enum PoseRenderer {

    // Editable limb table: (jointA, jointB, RGB color). COCO-style limbs.
    private static let limbs: [(VNHumanBodyPoseObservation.JointName,
                                VNHumanBodyPoseObservation.JointName,
                                (CGFloat, CGFloat, CGFloat))] = [
        (.neck, .nose,          (1.00, 0.00, 0.33)),
        (.neck, .leftShoulder,  (1.00, 0.33, 0.00)),
        (.neck, .rightShoulder, (1.00, 0.67, 0.00)),
        (.leftShoulder, .leftElbow,   (0.67, 1.00, 0.00)),
        (.leftElbow, .leftWrist,      (0.33, 1.00, 0.00)),
        (.rightShoulder, .rightElbow, (0.00, 1.00, 0.33)),
        (.rightElbow, .rightWrist,    (0.00, 1.00, 0.67)),
        (.neck, .leftHip,   (0.00, 0.67, 1.00)),
        (.neck, .rightHip,  (0.00, 0.33, 1.00)),
        (.leftHip, .leftKnee,    (0.33, 0.00, 1.00)),
        (.leftKnee, .leftAnkle,  (0.67, 0.00, 1.00)),
        (.rightHip, .rightKnee,  (1.00, 0.00, 1.00)),
        (.rightKnee, .rightAnkle,(1.00, 0.00, 0.67)),
        (.nose, .leftEye,   (0.80, 0.80, 0.00)),
        (.nose, .rightEye,  (0.00, 0.80, 0.80)),
        (.leftEye, .leftEar,   (0.80, 0.40, 0.00)),
        (.rightEye, .rightEar, (0.00, 0.40, 0.80)),
        (.leftShoulder, .rightShoulder, (0.90, 0.90, 0.90)),
        (.leftHip, .rightHip,           (0.00, 0.90, 0.90)),
    ]

    private static let confidenceThreshold: Float = 0.1

    static func skeleton(from pixelBuffer: CVPixelBuffer,
                         width: Int, height: Int) throws -> CGImage {
        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])
        let observations = request.results ?? []

        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw ControlProcessingError.writeFailed }

        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setLineCap(.round)
        ctx.setLineWidth(max(4, CGFloat(min(width, height)) / 120))  // thick lines

        for obs in observations {
            let points = (try? obs.recognizedPoints(.all)) ?? [:]
            func pt(_ name: VNHumanBodyPoseObservation.JointName) -> CGPoint? {
                guard let p = points[name], p.confidence >= confidenceThreshold else { return nil }
                // Vision normalized coords are bottom-left origin; CGContext too.
                return CGPoint(x: p.location.x * CGFloat(width),
                               y: p.location.y * CGFloat(height))
            }
            for (a, b, rgb) in limbs {
                guard let pa = pt(a), let pb = pt(b) else { continue }
                ctx.setStrokeColor(CGColor(red: rgb.0, green: rgb.1, blue: rgb.2, alpha: 1))
                ctx.move(to: pa); ctx.addLine(to: pb); ctx.strokePath()
            }
        }
        guard let cg = ctx.makeImage() else { throw ControlProcessingError.writeFailed }
        return cg
    }
}
