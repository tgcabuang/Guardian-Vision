#define MyAppName "Guardian Vision v3"
#define MyAppVersion "3.0.0"
#define MyAppPublisher "Guardian Vision v3"
#define MyAppExeName "Guardian Vision.exe"

[Setup]
AppId={{9B2C4D6E-3456-4F8A-ABCD-123456789223}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Guardian Vision v3
DefaultGroupName=Guardian Vision v3
DisableDirPage=no
DisableProgramGroupPage=yes
OutputDir=D:\GV_Revision_v3\installer_output
OutputBaseFilename=Guardian Vision v3 Setup
Compression=lzma2
SolidCompression=no
ArchitecturesInstallIn64BitMode=x64
SetupIconFile=D:\GV_Revision_v3\image\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName=Guardian Vision v3
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "D:\GV_Revision_v3\dist\win-unpacked\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Guardian Vision v3"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Guardian Vision v3"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Guardian Vision v3"; Flags: nowait postinstall skipifsilent
