import Foundation
import Combine

/// HTTP + WebSocket client for communicating with the Python FastAPI backend.
class BackendService: ObservableObject {
    private let baseURL = "http://127.0.0.1:8000"
    private let wsBaseURL = "ws://127.0.0.1:8000"
    private let session = URLSession.shared
    private let decoder = JSONDecoder()

    // MARK: - System Endpoints

    func healthCheck() async throws -> HealthResponse {
        return try await get("/api/v1/system/health")
    }

    func systemInfo() async throws -> SystemInfoResponse {
        return try await get("/api/v1/system/info")
    }

    func memoryStats() async throws -> MemoryStats {
        return try await get("/api/v1/system/memory")
    }

    // MARK: - Generation

    func generateTextToVideo(request: T2VRequest) async throws -> JobResponse {
        return try await post("/api/v1/generate/text-to-video", body: request)
    }

    func generatePreview(request: PreviewRequest) async throws -> JobResponse {
        return try await post("/api/v1/generate/preview", body: request)
    }

    func generateImageToVideo(request: I2VRequest) async throws -> JobResponse {
        return try await post("/api/v1/generate/image-to-video", body: request)
    }

    func getJobStatus(jobId: String) async throws -> JobStatus {
        return try await get("/api/v1/queue/\(jobId)")
    }

    func cancelJob(jobId: String) async throws -> CancelResponse {
        return try await post("/api/v1/queue/\(jobId)/cancel", body: EmptyBody())
    }

    // MARK: - Prompt Enhancement

    func enhancePrompt(prompt: String, isI2V: Bool = false) async throws -> EnhanceResponse {
        struct Body: Encodable {
            let prompt: String
            let is_i2v: Bool
        }
        return try await post("/api/v1/prompt/enhance", body: Body(prompt: prompt, is_i2v: isI2V))
    }

    // MARK: - Retake & Extend

    func generateRetake(request: RetakeRequest) async throws -> JobResponse {
        return try await post("/api/v1/generate/retake", body: request)
    }

    func generateExtend(request: ExtendRequest) async throws -> JobResponse {
        return try await post("/api/v1/generate/extend", body: request)
    }

    // MARK: - WebSocket Progress

    func connectProgress(jobId: String) -> AsyncStream<ProgressUpdate> {
        AsyncStream { continuation in
            let url = URL(string: "\(wsBaseURL)/ws/progress/\(jobId)")!
            let task = session.webSocketTask(with: url)
            task.resume()

            func receiveNext() {
                task.receive { result in
                    switch result {
                    case .success(let message):
                        switch message {
                        case .string(let text):
                            if let data = text.data(using: .utf8),
                               let update = try? JSONDecoder().decode(ProgressUpdate.self, from: data) {
                                continuation.yield(update)
                                if update.done == true {
                                    continuation.finish()
                                    task.cancel(with: .normalClosure, reason: nil)
                                    return
                                }
                            }
                        default:
                            break
                        }
                        receiveNext()
                    case .failure:
                        continuation.finish()
                    }
                }
            }

            receiveNext()

            continuation.onTermination = { _ in
                task.cancel(with: .normalClosure, reason: nil)
            }
        }
    }

    // MARK: - LoRA Management

    func listLoRAs() async throws -> [LoRAInfo] {
        return try await get("/api/v1/loras")
    }

    @discardableResult
    func loadLoRA(loraId: String) async throws -> SuccessResponse {
        struct Body: Encodable { let lora_id: String }
        return try await post("/api/v1/loras/load", body: Body(lora_id: loraId))
    }

    @discardableResult
    func unloadLoRA(loraId: String) async throws -> SuccessResponse {
        return try await post("/api/v1/loras/unload/\(loraId)", body: EmptyBody())
    }

    // MARK: - Export

    func exportVideo(request: ExportVideoRequest) async throws -> PathResponse {
        return try await post("/api/v1/export/video", body: request)
    }

    func exportFCPXML(request: ExportFCPXMLRequest) async throws -> PathResponse {
        return try await post("/api/v1/export/fcpxml", body: request)
    }

    // MARK: - HTTP Helpers

    private func get<T: Decodable>(_ path: String) async throws -> T {
        let url = URL(string: "\(baseURL)\(path)")!
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            let body = String(data: data, encoding: .utf8) ?? "(no body)"
            throw BackendError.requestFailed(code, body)
        }
        return try decoder.decode(T.self, from: data)
    }

    private func post<B: Encodable, T: Decodable>(_ path: String, body: B) async throws -> T {
        let url = URL(string: "\(baseURL)\(path)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            let body = String(data: data, encoding: .utf8) ?? "(no body)"
            throw BackendError.requestFailed(code, body)
        }
        return try decoder.decode(T.self, from: data)
    }
}

struct SuccessResponse: Decodable {
    let success: Bool
    let message: String
}

enum BackendError: LocalizedError {
    case requestFailed(Int, String)

    var errorDescription: String? {
        switch self {
        case .requestFailed(let code, let body): return "HTTP \(code): \(body)"
        }
    }
}

private struct EmptyBody: Encodable {}
