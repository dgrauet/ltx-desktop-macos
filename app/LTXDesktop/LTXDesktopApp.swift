import SwiftUI

@main
struct LTXDesktopApp: App {
    @StateObject private var processManager = ProcessManager()
    @StateObject private var backendService = BackendService()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(processManager)
                .environmentObject(backendService)
                .onAppear {
                    processManager.startBackend()
                }
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 1200, height: 800)
    }
}
