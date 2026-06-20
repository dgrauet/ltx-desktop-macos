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

        // Kill any stale process already holding port 8000
        let kill = Process()
        kill.executableURL = URL(fileURLWithPath: "/bin/sh")
        kill.arguments = ["-c", "lsof -ti :8000 | xargs kill -9 2>/dev/null; true"]
        try? kill.run()
        kill.waitUntilExit()

        guard let backendDir = findBackendDir() else {
            lastError = "Backend directory not found. Make sure backend/main.py exists in the project."
            return
        }

        guard let pythonURL = findPythonExecutable(for: backendDir) else {
            lastError = "Python runtime not found. Run scripts/setup.sh (dev) or reinstall the app (release)."
            return
        }

        let proc = Process()
        proc.executableURL = pythonURL
        proc.arguments = ["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
        proc.currentDirectoryURL = backendDir
        proc.environment = backendEnvironment(for: backendDir)

        // Pipe stdout/stderr and read asynchronously so the buffer never fills
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe

        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let text = String(data: data, encoding: .utf8) {
                // Log to console only — don't surface backend stderr as UI errors
                NSLog("[Backend] %@", text.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }

        proc.terminationHandler = { [weak self] process in
            pipe.fileHandleForReading.readabilityHandler = nil
            DispatchQueue.main.async {
                self?.isBackendReady = false
                if process.terminationStatus != 0 && process.terminationStatus != 15 {
                    let current = self?.lastError ?? ""
                    self?.lastError = "Backend crashed (exit \(process.terminationStatus)): \(current)"
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

    private func bundledResourcesURL() -> URL? {
        Bundle.main.resourceURL
    }

    private func findBackendDir() -> URL? {
        // Release layout: Contents/Resources/backend
        if let resources = bundledResourcesURL() {
            let bundled = resources.appendingPathComponent("backend")
            if FileManager.default.fileExists(atPath: bundled.appendingPathComponent("main.py").path) {
                return bundled
            }
        }

        // Dev layout: walk up from compile-time source path
        let sourceFile = URL(fileURLWithPath: #file)
        let projectRoot = sourceFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let candidate = projectRoot.appendingPathComponent("backend")
        if FileManager.default.fileExists(atPath: candidate.appendingPathComponent("main.py").path) {
            return candidate
        }

        // Legacy layout: sibling of the .app bundle
        let bundleSibling = URL(fileURLWithPath: Bundle.main.bundlePath)
            .deletingLastPathComponent()
            .appendingPathComponent("backend")
        if FileManager.default.fileExists(atPath: bundleSibling.appendingPathComponent("main.py").path) {
            return bundleSibling
        }
        return nil
    }

    private func findPythonExecutable(for backendDir: URL) -> URL? {
        if let resources = bundledResourcesURL(),
           backendDir.path.hasPrefix(resources.path) {
            let bundledPython = resources.appendingPathComponent("python/bin/python3")
            if FileManager.default.fileExists(atPath: bundledPython.path) {
                return bundledPython
            }
        }

        let venvPython = backendDir.appendingPathComponent(".venv/bin/python")
        if FileManager.default.fileExists(atPath: venvPython.path) {
            return venvPython
        }
        return nil
    }

    private func backendEnvironment(for backendDir: URL) -> [String: String] {
        var env = ProcessInfo.processInfo.environment

        if let resources = bundledResourcesURL(),
           backendDir.path.hasPrefix(resources.path) {
            let binDir = resources.appendingPathComponent("bin").path
            let existingPath = env["PATH"] ?? ""
            env["PATH"] = existingPath.isEmpty ? binDir : "\(binDir):\(existingPath)"
            env["LTX_PYTHON"] = resources.appendingPathComponent("python/bin/python3").path
            env["LTX_FFMPEG_PATH"] = resources.appendingPathComponent("bin/ffmpeg").path
            env["LTX_FFPROBE_PATH"] = resources.appendingPathComponent("bin/ffprobe").path
        } else {
            for extra in ["/opt/homebrew/bin", "/usr/local/bin"] {
                let existingPath = env["PATH"] ?? ""
                if !existingPath.contains(extra) {
                    env["PATH"] = existingPath.isEmpty ? extra : "\(extra):\(existingPath)"
                }
            }
        }

        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env
    }

    deinit {
        stopBackend()
    }
}
