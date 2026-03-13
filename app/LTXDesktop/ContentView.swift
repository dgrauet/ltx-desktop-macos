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
        HStack(spacing: 0) {
            // Sidebar
            VStack(spacing: 0) {
                Text("LTX Desktop")
                    .font(.headline)
                    .padding(.vertical, 12)
                    .frame(maxWidth: .infinity)

                Divider()

                ForEach(Tab.allCases, id: \.self) { tab in
                    sidebarButton(tab)
                }

                Spacer()
            }
            .frame(width: 180)
            .background(.ultraThinMaterial)

            Divider()

            // Detail
            Group {
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
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func sidebarButton(_ tab: Tab) -> some View {
        Button {
            selectedTab = tab
        } label: {
            Label(tab.rawValue, systemImage: iconForTab(tab))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(
            selectedTab == tab
                ? Color.accentColor.opacity(0.15)
                : Color.clear
        )
        .foregroundStyle(selectedTab == tab ? .primary : .secondary)
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
