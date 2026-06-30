import SwiftUI

struct ContentView: View {
    @EnvironmentObject var processManager: ProcessManager
    @EnvironmentObject var backendService: BackendService
    @StateObject private var queueVM = QueueViewModel()
    @StateObject private var generationVM = GenerationViewModel()
    @StateObject private var trainingVM = TrainingViewModel()

    enum Tab: String, CaseIterable {
        case generation = "Generation"
        case queue = "Queue"
        case history = "History"
        case controlVideos = "Control Videos"
        case lora = "LoRA"
        case training = "Training"
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

            // Detail + persistent progress bar
            VStack(spacing: 0) {
                Group {
                    if !processManager.isBackendReady {
                        preparingView
                    } else {
                        switch selectedTab {
                        case .generation:
                            GenerationView()
                        case .queue:
                            QueueView()
                        case .history:
                            HistoryView()
                        case .controlVideos:
                            ControlLibraryView(onUse: { selectedTab = .generation })
                        case .lora:
                            LoRAView()
                        case .training:
                            ScrollView {
                                VStack(spacing: 0) {
                                    trainingBetaBanner
                                    DatasetBuilderView()
                                    Divider()
                                    TrainingConfigView()
                                    Divider()
                                    TrainingRunView()
                                }
                            }
                        case .settings:
                            SettingsView()
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                // Persistent progress bar for running jobs
                if let job = queueVM.runningJob {
                    activeJobBar(job)
                }
            }
        }
        .onAppear {
            if processManager.isBackendReady {
                queueVM.startPolling(service: backendService)
            }
        }
        .onChange(of: processManager.isBackendReady) { _, ready in
            if ready {
                queueVM.startPolling(service: backendService)
            } else {
                queueVM.stopPolling()
            }
        }
        .environmentObject(generationVM)
        .environmentObject(trainingVM)
    }

    private var trainingBetaBanner: some View {
        HStack(spacing: 8) {
            Text("BETA")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.purple)
                .clipShape(Capsule())
            Text("LoRA training is in beta and under active testing — results and stability may vary.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.purple.opacity(0.12))
    }

    private func sidebarButton(_ tab: Tab) -> some View {
        Button {
            selectedTab = tab
        } label: {
            HStack {
                Label(tab.rawValue, systemImage: iconForTab(tab))
                Spacer()
                if tab == .queue, queueVM.activeCount > 0 {
                    Text("\(queueVM.activeCount)")
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(queueVM.runningCount > 0 ? Color.green : Color.orange)
                        .clipShape(Capsule())
                }
                if tab == .training {
                    Text("BETA")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.purple)
                        .clipShape(Capsule())
                        .help("Training is in beta — still under testing")
                }
            }
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

    // MARK: - Active Job Progress Bar

    private func activeJobBar(_ job: QueueEntry) -> some View {
        HStack(spacing: 10) {
            // Pulsing indicator
            Circle()
                .fill(.green)
                .frame(width: 8, height: 8)

            // Job type badge
            Text(job.jobType.uppercased())
                .font(.caption2)
                .fontWeight(.bold)
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(Color.accentColor.opacity(0.2))
                .clipShape(RoundedRectangle(cornerRadius: 3))

            // Prompt (truncated)
            Text(job.prompt)
                .font(.caption)
                .lineLimit(1)
                .foregroundStyle(.secondary)
                .frame(maxWidth: 200, alignment: .leading)

            // Progress bar
            ProgressView(value: job.progress ?? 0)
                .frame(maxWidth: 160)

            // Percentage
            Text("\(Int((job.progress ?? 0) * 100))%")
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 35, alignment: .trailing)

            // Status
            if let status = job.status {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            // Queue count
            if queueVM.queuedCount > 0 {
                Text("+\(queueVM.queuedCount) queued")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }

            // View in Queue button
            Button {
                selectedTab = .queue
            } label: {
                Image(systemName: "list.number")
                    .font(.caption)
            }
            .buttonStyle(.borderless)
            .help("View in Queue")

            // Cancel button
            Button {
                Task { await queueVM.cancelJob(jobId: job.jobId, service: backendService) }
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.red.opacity(0.7))
            }
            .buttonStyle(.borderless)
            .help("Cancel job")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.green.opacity(0.05))
        .overlay(alignment: .top) { Divider() }
    }

    private func iconForTab(_ tab: Tab) -> String {
        switch tab {
        case .generation: return "wand.and.sparkles"
        case .queue: return "list.number"
        case .history: return "clock.arrow.circlepath"
        case .controlVideos: return "square.stack.3d.up"
        case .lora: return "puzzlepiece.extension"
        case .training: return "brain.head.profile"
        case .settings: return "gear"
        }
    }
}
