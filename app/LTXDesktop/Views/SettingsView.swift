import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var backendService: BackendService
    @StateObject private var memoryVM = MemoryViewModel()

    var body: some View {
        TabView {
            memoryTab
                .tabItem {
                    Label("Memory", systemImage: "memorychip")
                }
        }
        .padding()
        .onAppear {
            memoryVM.startPolling(service: backendService, isGenerating: false)
        }
        .onDisappear {
            memoryVM.stopPolling()
        }
    }

    // MARK: - Memory Tab

    private var memoryTab: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Metal Memory Monitor")
                .font(.title2)
                .fontWeight(.semibold)

            if let stats = memoryVM.memoryStats {
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible()),
                ], spacing: 16) {
                    memoryCard(
                        title: "Active Memory",
                        value: stats.activeMemoryGb,
                        color: .blue
                    )
                    memoryCard(
                        title: "Cache Memory",
                        value: stats.cacheMemoryGb,
                        color: memoryVM.highCacheWarning ? .yellow : .green
                    )
                    memoryCard(
                        title: "Peak Memory",
                        value: stats.peakMemoryGb,
                        color: memoryVM.criticalPeakWarning ? .red : .orange
                    )
                    memoryCard(
                        title: "System Available",
                        value: stats.systemAvailableGb,
                        color: memoryVM.lowAvailableWarning ? .red : .mint
                    )
                }

                Divider()

                // Reload counter
                HStack {
                    Text("Generations since reload:")
                        .foregroundStyle(.secondary)
                    Text("\(stats.generationCountSinceReload) / 5")
                        .fontWeight(.medium)
                        .monospacedDigit()
                    Spacer()
                    Text("Next reload in \(stats.nextReloadIn) generations")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                // Warnings
                if memoryVM.highCacheWarning {
                    warningBanner(
                        message: "High cache — cleanup recommended",
                        color: .yellow
                    )
                }
                if memoryVM.criticalPeakWarning {
                    warningBanner(
                        message: "Memory critical — peak exceeds 85% of RAM",
                        color: .red
                    )
                }
                if memoryVM.lowAvailableWarning {
                    warningBanner(
                        message: "Low memory — reduce resolution or frame count",
                        color: .red
                    )
                }
            } else {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Loading memory stats...")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            Spacer()
        }
        .padding()
    }

    // MARK: - Components

    private func memoryCard(title: String, value: Double, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(String(format: "%.2f GB", value))
                .font(.title3)
                .fontWeight(.semibold)
                .monospacedDigit()
            Gauge(value: min(value, 64), in: 0...64) {
                EmptyView()
            }
            .tint(color)
        }
        .padding(12)
        .background(Color(.controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func warningBanner(message: String, color: Color) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(color)
            Text(message)
                .font(.callout)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
