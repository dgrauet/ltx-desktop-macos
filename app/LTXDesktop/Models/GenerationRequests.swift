import Foundation

struct RetakeRequest: Encodable {
    let sourceVideoPath: String
    let prompt: String
    let startTimeS: Double
    let endTimeS: Double
    let steps: Int
    let seed: Int
    let fps: Int

    enum CodingKeys: String, CodingKey {
        case sourceVideoPath = "source_video_path"
        case prompt, steps, seed, fps
        case startTimeS = "start_time_s"
        case endTimeS = "end_time_s"
    }
}

struct ExtendRequest: Encodable {
    let sourceVideoPath: String
    let prompt: String
    let direction: String
    let extensionFrames: Int
    let steps: Int
    let seed: Int
    let fps: Int

    enum CodingKeys: String, CodingKey {
        case sourceVideoPath = "source_video_path"
        case prompt, direction, steps, seed, fps
        case extensionFrames = "extension_frames"
    }
}
