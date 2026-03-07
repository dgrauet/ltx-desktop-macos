import Foundation
import Combine

/// Manages the Python backend subprocess lifecycle.
class ProcessManager: ObservableObject {
    @Published var isBackendReady = false
    @Published var lastError: String?

    private var process: Process?
    private var healthCheckTimer: Timer?
    private let backendURL = "http://127.0.0.1:8000"

    func startBackend() {
        guard process == nil else { return }

        // Find project root (app/ is one level inside)
        let appDir = Bundle.main.bundlePath
        let projectRoot = URL(fileURLWithPath: appDir)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let backendDir = projectRoot.appendingPathComponent("backend")

        // Find uv binary
        let uvPaths = [
            "\(NSHomeDirectory())/.local/bin/uv",
            "/opt/homebrew/bin/uv",
            "/usr/local/bin/uv",
        ]
        guard let uvPath = uvPaths.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
            lastError = "uv not found. Run scripts/setup.sh first."
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: uvPath)
        proc.arguments = ["run", "--prerelease=allow", "python", "-m", "uvicorn",
                          "main:app", "--host", "127.0.0.1", "--port", "8000"]
        proc.currentDirectoryURL = backendDir
        proc.environment = ProcessInfo.processInfo.environment

        // Pipe stdout/stderr
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe

        proc.terminationHandler = { [weak self] process in
            DispatchQueue.main.async {
                self?.isBackendReady = false
                if process.terminationStatus != 0 && process.terminationStatus != 15 {
                    self?.lastError = "Backend crashed (exit code \(process.terminationStatus))"
                }
            }
        }

        do {
            try proc.run()
            self.process = proc
            startHealthCheck()
        } catch {
            lastError = "Failed to start backend: \(error.localizedDescription)"
        }
    }

    func stopBackend() {
        healthCheckTimer?.invalidate()
        healthCheckTimer = nil

        guard let proc = process, proc.isRunning else {
            process = nil
            return
        }

        proc.terminate()

        // Wait up to 5 seconds, then force kill
        DispatchQueue.global().asyncAfter(deadline: .now() + 5) { [weak self] in
            if proc.isRunning {
                proc.interrupt()
            }
            DispatchQueue.main.async {
                self?.process = nil
                self?.isBackendReady = false
            }
        }
    }

    func restartBackend() {
        lastError = nil
        stopBackend()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
            self?.startBackend()
        }
    }

    private func startHealthCheck() {
        healthCheckTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] timer in
            guard let self = self else { timer.invalidate(); return }

            let url = URL(string: "\(self.backendURL)/api/v1/system/health")!
            URLSession.shared.dataTask(with: url) { data, response, error in
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    return
                }
                DispatchQueue.main.async {
                    timer.invalidate()
                    self.healthCheckTimer = nil
                    self.isBackendReady = true
                    self.lastError = nil
                }
            }.resume()
        }
    }

    deinit {
        stopBackend()
    }
}
