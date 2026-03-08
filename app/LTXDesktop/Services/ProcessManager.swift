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

        // Walk up from bundle path to find project root containing backend/main.py.
        // This works both in DerivedData (dev) and when the app is bundled.
        guard let backendDir = findBackendDir() else {
            lastError = "Backend directory not found. Make sure backend/main.py exists in the project."
            return
        }

        // Use the venv Python directly to avoid uv re-syncing the lockfile
        // (uv run would revert manually pip-installed packages to lockfile versions)
        let venvPython = backendDir.appendingPathComponent(".venv/bin/python").path
        guard FileManager.default.fileExists(atPath: venvPython) else {
            lastError = "Python venv not found at \(venvPython). Run scripts/setup.sh first."
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: venvPython)
        proc.arguments = ["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
        proc.currentDirectoryURL = backendDir
        proc.environment = ProcessInfo.processInfo.environment

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

    private func findBackendDir() -> URL? {
        // Primary: use compile-time source path (#file = …/app/LTXDesktop/Services/ProcessManager.swift)
        // Going up 4 levels: Services → LTXDesktop → app → project root
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
        // Fallback: check a sibling of the .app bundle (distribution layout)
        let bundleSibling = URL(fileURLWithPath: Bundle.main.bundlePath)
            .deletingLastPathComponent()
            .appendingPathComponent("backend")
        if FileManager.default.fileExists(atPath: bundleSibling.appendingPathComponent("main.py").path) {
            return bundleSibling
        }
        return nil
    }

    deinit {
        stopBackend()
    }
}
