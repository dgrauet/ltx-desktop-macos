import SwiftUI

struct ContentView: View {
    @EnvironmentObject var processManager: ProcessManager
    @EnvironmentObject var backendService: BackendService

    enum Tab: String, CaseIterable {
        case generation = "Generation"
        case history = "History"
        case lora = "LoRA"
        case settings = "Settings"
    }

    @State private var selectedTab: Tab = .generation

    var body: some View {
        NavigationSplitView {
            List(Tab.allCases, id: \.self, selection: $selectedTab) { tab in
                Label(tab.rawValue, systemImage: iconForTab(tab))
            }
            .listStyle(.sidebar)
            .navigationTitle("LTX Desktop")
        } detail: {
            if !processManager.isBackendReady {
                preparingView
            } else {
                switch selectedTab {
                case .generation:
                    GenerationView()
                case .history:
                    HistoryView()
                case .lora:
                    LoRAView()
                case .settings:
                    SettingsView()
                }
            }
        }
    }

    private var preparingView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.5)
            Text("Preparing engine...")
                .font(.title2)
                .foregroundStyle(.secondary)
            if let error = processManager.lastError {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.caption)
                Button("Restart Backend") {
                    processManager.restartBackend()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func iconForTab(_ tab: Tab) -> String {
        switch tab {
        case .generation: return "wand.and.sparkles"
        case .history: return "clock.arrow.circlepath"
        case .lora: return "puzzlepiece.extension"
        case .settings: return "gear"
        }
    }
}
