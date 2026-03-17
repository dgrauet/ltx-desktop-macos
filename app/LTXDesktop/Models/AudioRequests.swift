import Foundation

struct TTSRequest: Encodable {
    let text: String
    let voice: String
    let speed: Double
}

struct MusicRequest: Encodable {
    let genre: String
    let duration: Double
}

struct AudioMixRequest: Encodable {
    let videoPath: String
    let ttsPath: String?
    let musicPath: String?
    let ttsVolume: Double
    let musicVolume: Double
    let videoAudioVolume: Double

    enum CodingKeys: String, CodingKey {
        case videoPath = "video_path"
        case ttsPath = "tts_path"
        case musicPath = "music_path"
        case ttsVolume = "tts_volume"
        case musicVolume = "music_volume"
        case videoAudioVolume = "video_audio_volume"
    }
}
